
"""
CRDS helpers for retrieving existing Roman WFI reference files.
 
Suggested location: src/wfi_reference_pipeline/utilities/crds_utils.py
 
Generic over reference type so the same code serves MASK (ReferenceType
get_crds_mask), and GAIN / AREA (Photom), or any other reftype.
 
Requires CRDS_SERVER_URL and CRDS_PATH to be set in the environment.
"""
 
import logging
import os
import subprocess
 
import numpy as np
import roman_datamodels as rdm
import crds
from crds.client import api
from crds.core import config as crds_config
 
CRDS_OBSERVATORY = "roman"
CRDS_INSTRUMENT = "wfi"
N_WFI_DETECTORS = 18
 
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
 
 
def fetch_crds_reference_files(reftype, crds_context=None, refresh=False):
    """
    Ensure every file of `reftype` in a CRDS context is on disk and return
    a {detector: local_path} dictionary.
 
    The first call syncs the context, checks which files are already in
    CRDS_PATH, downloads only the missing ones in a single request, and
    caches the result for the rest of the process.
 
    Parameters
    ----------
    reftype : str
        CRDS reference type, e.g. 'mask', 'gain', 'area'.
    crds_context : str or None, optional
        CRDS context, e.g. 'roman_0042.pmap'. Defaults to latest.
    refresh : bool, optional
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
 
    # 1. Sync mappings before downloading
    sync_crds_context(crds_context)
 
    # 2. Every file named in this context's rmap for the reftype
    ref_names = (
        crds.rmap.load_mapping(crds_context)
        .get_imap(CRDS_INSTRUMENT)
        .get_rmap(reftype)
        .reference_names()
    )
    if not ref_names:
        raise RuntimeError(f"No '{reftype}' reference files found in {crds_context}")
 
    # 3. Which ones are already downloaded?
    local_paths, missing = {}, []
    for name in ref_names:
        path = crds_config.locate_file(name, CRDS_OBSERVATORY)
        if os.path.exists(path):
            local_paths[name] = path
        else:
            missing.append(name)
 
    logging.info(
        f"CRDS {reftype}: {len(local_paths)} of {len(ref_names)} files cached; "
        f"downloading {len(missing)}."
    )
 
    # 4. Download only what's missing, in one request
    if missing:
        local_paths.update(api.dump_references(crds_context, baserefs=missing))
 
    # 5. Map detector -> file. If an rmap lists several files for one detector
    #    (different USEAFTER dates), keep the most recent.
    by_detector = {}
    for path in local_paths.values():
        try:
            with rdm.open(path) as ref:
                det = str(ref.meta.instrument.detector)
                useafter = ref.meta.useafter
        except Exception as err:
            logging.warning(f"Could not read CRDS {reftype} file {path}: {err}")
            continue
        if det not in by_detector or useafter > by_detector[det][1]:
            by_detector[det] = (path, useafter)
 
    files = {det: path for det, (path, _) in sorted(by_detector.items())}
 
    if len(files) != N_WFI_DETECTORS:
        logging.warning(
            f"CRDS context {crds_context} has {reftype} files for {len(files)} "
            f"of {N_WFI_DETECTORS} detectors: {sorted(files)}"
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
