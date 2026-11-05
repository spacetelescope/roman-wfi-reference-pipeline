"""
CRDS helpers for retrieving existing Roman WFI reference files.
 
Generic over reference type, so the same code serves MASK (ReferenceType
get_crds_mask) and GAIN / AREA (Photom), or any reftype whose rmap selects
on detector, exposure type and USEAFTER. Exposure type is assumed to be
imaging mode (WFI_IMAGE) for now.
 
For each detector, CRDS's own rmap rules choose the file that applies now:
the file with the latest USEAFTER at or before the current time. Only those
files (at most one per detector) are downloaded.
 
Requires CRDS_SERVER_URL and CRDS_PATH to be set in the environment.
"""
 
import logging
import os
import subprocess
 
import crds
import numpy as np
import roman_datamodels as rdm
from astropy.time import Time
from crds.client import api
from crds.core import config as crds_config
 
from wfi_reference_pipeline.constants import WFI_DETECTORS, WFI_TYPE_IMAGE
 
CRDS_OBSERVATORY = "roman"
CRDS_INSTRUMENT = "wfi"
 
# rmap selection keys this module knows how to fill in.
CRDS_DETECTOR_KEY = "ROMAN.META.INSTRUMENT.DETECTOR"
CRDS_START_TIME_KEY = "ROMAN.META.EXPOSURE.START_TIME"
CRDS_EXPOSURE_TYPE_KEY = "ROMAN.META.EXPOSURE.TYPE"
 
# All selections currently assume imaging mode.
CRDS_EXPOSURE_TYPE = WFI_TYPE_IMAGE
SUPPORTED_PARKEYS = {CRDS_DETECTOR_KEY, CRDS_START_TIME_KEY, CRDS_EXPOSURE_TYPE_KEY}
 
# Process-wide caches: one sync per context, one bulk download per
# (context, reftype). Instances for WFI01..WFI18 reuse the same lookup.
_SYNCED_CONTEXTS = set()
_REFERENCE_FILES = {}  # {(crds_context, reftype): {"WFI01": "/path/file.asdf", ...}}
 
 
def get_crds_context(crds_context=None):
    """Return crds_context, or the latest operational Roman context if None."""
    if crds_context is None:
        crds_context = crds.get_default_context(observatory=CRDS_OBSERVATORY, state="latest")
    return crds_context
 
 
def sync_crds_context(crds_context):
    """
    Sync CRDS mappings for one context, once per process.
 
    Uses `crds sync --contexts <ctx>` rather than `--all`, which pulls
    mappings for every historical context.
    """
    if crds_context in _SYNCED_CONTEXTS:
        return
 
    logging.info(f"Syncing CRDS mappings for context {crds_context}")
    try:
        result = subprocess.run(
            ["crds", "sync", "--contexts", crds_context],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.debug(result.stdout)
        _SYNCED_CONTEXTS.add(crds_context)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        stderr = getattr(err, "stderr", None) or str(err)
        logging.warning(
            f"crds sync failed for {crds_context}: {stderr.strip()} "
            "Continuing with the local CRDS cache / server."
        )
 
 
def select_crds_reference_files(reftype, crds_context=None):
    """
    Ask CRDS which file of `reftype` applies to each detector right now.
 
    Uses the rmap's own selection rules, so for every detector this returns
    the file with the latest USEAFTER at or before the current time. A file
    whose USEAFTER is in the future is not selected. Nothing is downloaded.
 
    Selection assumes imaging mode: for rmaps that also select on exposure
    type (e.g. DARK, READNOISE), ROMAN.META.EXPOSURE.TYPE is set to
    WFI_IMAGE.
 
    Parameters
    ----------
    reftype: str
        CRDS reference type, e.g. 'mask', 'gain', 'area'.
    crds_context: str or None, optional
        CRDS context, e.g. 'roman_0042.pmap'. Defaults to latest.
 
    Returns
    -------
    dict
        {detector_name: reference_file_basename}, e.g.
        {'WFI07': 'roman_wfi_mask_0070.asdf', ...}. Detectors with no
        applicable file are left out, with a warning.
 
    Raises
    ------
    ValueError
        If the reftype's rmap selects on anything other than detector,
        start time and exposure type (e.g. FLAT also selects on optical
        element), since this function cannot fill in those values.
    """
    crds_context = get_crds_context(crds_context)
    sync_crds_context(crds_context)
 
    rmap = (
        crds.rmap.load_mapping(crds_context)
        .get_imap(CRDS_INSTRUMENT)
        .get_rmap(reftype.lower())
    )
 
    required = set(rmap.get_required_parkeys())
    unsupported = required - SUPPORTED_PARKEYS
    if unsupported:
        raise ValueError(
            f"The {reftype} rmap also selects on {sorted(unsupported)}; "
            "select_crds_reference_files only supports detector, start time "
            "and exposure type."
        )
 
    # CRDS standard date form, e.g. '2026-10-21 16:12:00'
    now = Time.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Selecting CRDS {reftype} files in {crds_context} for {now} UTC")
 
    selected = {}
    for detector in sorted(WFI_DETECTORS):
        header = {
            CRDS_DETECTOR_KEY: detector,
            CRDS_START_TIME_KEY: now,
            CRDS_EXPOSURE_TYPE_KEY: CRDS_EXPOSURE_TYPE,
        }
        # get_best_ref returns a basename, None, or a "NOT FOUND ..." string
        bestref = rmap.get_best_ref(header)
        if not bestref or bestref.startswith("NOT FOUND"):
            logging.warning(f"No CRDS {reftype} file for {detector} in {crds_context}: {bestref}")
            continue
        selected[detector] = bestref
 
    return selected
 
 
def fetch_crds_reference_files(reftype, crds_context=None, refresh=False):
    """
    Download the CRDS `reftype` file that applies now for each detector and
    return a {detector: local_path} dictionary.
 
    The first call syncs the context, asks CRDS which file applies to each
    detector (see `select_crds_reference_files`), checks which of those are
    already in CRDS_PATH, downloads only the missing ones in a single request,
    and caches the result for the rest of the process.
 
    Parameters
    ----------
    reftype: str
        CRDS reference type, e.g. 'mask', 'gain', 'area'.
    crds_context: str or None, optional
        CRDS context, e.g. 'roman_0042.pmap'. Defaults to latest.
    refresh: bool, optional
        If True, ignore the in-process cache and re-check CRDS.
 
    Returns
    -------
    dict
        {detector_name: local_file_path}
    """
    reftype = reftype.lower()
    crds_context = get_crds_context(crds_context)
    key = (crds_context, reftype)
 
    if not refresh and key in _REFERENCE_FILES:
        return _REFERENCE_FILES[key]
 
    # 1. Sync, then let CRDS pick one file per detector for the current time
    selected = select_crds_reference_files(reftype, crds_context)
 
    # 2. Which of the selected files are already downloaded?
    local_paths, missing = {}, []
    for name in sorted(set(selected.values())):
        path = crds_config.locate_file(name, CRDS_OBSERVATORY)
        if os.path.exists(path):
            local_paths[name] = path
        else:
            missing.append(name)
 
    logging.info(
        f"CRDS {reftype}: {len(selected)} files selected, {len(local_paths)} already "
        f"cached; downloading {len(missing)}."
    )
 
    # 3. Download only what's missing, in one request
    if missing:
        local_paths.update(api.dump_references(crds_context, baserefs=missing))
 
    files = {det: local_paths[name] for det, name in sorted(selected.items())}
 
    if len(files) != len(WFI_DETECTORS):
        logging.warning(
            f"CRDS context {crds_context} has {reftype} files for {len(files)} "
            f"of {len(WFI_DETECTORS)} detectors: {sorted(files)}"
        )
 
    _REFERENCE_FILES[key] = files
    return files
 
 
def get_crds_reference_path(reftype, detector, crds_context=None):
    """Local path of the CRDS `reftype` file for one detector (e.g. 'WFI01')."""
    files = fetch_crds_reference_files(reftype, crds_context)
    if detector not in files:
        raise KeyError(
            f"No CRDS {reftype} file for detector {detector}. Available: {sorted(files)}"
        )
    return files[detector]
 
 
def load_crds_bad_pixel_map(detector, crds_context=None):
    """
    Return (dq array as uint32, local path) from the CRDS MASK file for a detector.
    """
    path = get_crds_reference_path("mask", detector, crds_context)
    with rdm.open(path) as ref:
        # np.array copies, so the data outlives the closed file
        bad_pixel_map = np.array(ref.dq, dtype=np.uint32)
    return bad_pixel_map, path
