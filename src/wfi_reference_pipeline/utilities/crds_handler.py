import logging
import os
import subprocess
 
import crds
import numpy as np
import roman_datamodels as rdm
from astropy.time import Time
from crds.client import api
from crds.core import config as crds_config

from wfi_reference_pipeline.constants import (
    CRDS_REF_TYPE_MAP,
    REF_TYPE_MASK,
    WFI_DETECTORS,
    WFI_TYPE_IMAGE,
    WFI_TYPES,
)

 
class CrdsHandler:
    """
    Handler for retrieving existing Roman WFI reference files from CRDS.
    Requires CRDS_SERVER_URL and CRDS_PATH to be set in the environment.

    Class to retrieve existing reference files from CRDS.
 
    Generic over reference type, so the same code serves MASK (Dark),
    GAIN / AREA (Photom) and GAIN / READNOISE / DARK / FLAT / SATURATION (ETC),
    or any reftype whose rmap selects on detector, exposure type, optical
    element and USEAFTER. Exposure type defaults to imaging mode (WFI_IMAGE)
    and can be set per handler. Optical element is passed in for rmaps that
    need it (e.g. FLAT).
 
    For each detector, CRDS's own rmap rules choose the file that applies now:
    the file with the latest USEAFTER at or before the current time. Only those
    files (at most one per detector) are downloaded.
 
    The handler uses the latest operational CRDS context unless a specific
    context is passed in. Syncs and downloads are cached at the class level,
    so every handler in a Python session with the same context reuses them
    (e.g. Dark instances for WFI01..WFI18 sync and download once).
 
    Usage:
        from wfi_reference_pipeline.utilities.crds_handler import CrdsHandler
        crds_handler = CrdsHandler()                            # latest context
        crds_handler = CrdsHandler(crds_context='roman_0042.pmap')  # specific context
        crds_handler = CrdsHandler(exposure_type=WFI_TYPE_GRISM)    # non-imaging exposure type
 
        # {detector: local_file_path} for all detectors
        gain_files = crds_handler.get_reference_files('gain')
        area_files = crds_handler.get_reference_files('area')   # or REF_TYPE_PIXELAREA
 
        # FLAT selects on optical element too, so name one
        flat_files = crds_handler.get_reference_files('flat', optical_element='F062')
 
        # Local path of one detector's file
        mask_file = crds_handler.get_reference_file('mask', 'WFI07')
 
        # uint32 DQ array and local path of one detector's CRDS MASK file
        bad_pixel_map, mask_file = crds_handler.get_bad_pixel_map('WFI07')
     """

    OBSERVATORY = "roman"
    INSTRUMENT = "wfi"
 
    # rmap selection keys the handler knows how to fill in.
    DETECTOR_KEY = "ROMAN.META.INSTRUMENT.DETECTOR"
    START_TIME_KEY = "ROMAN.META.EXPOSURE.START_TIME"
    EXPOSURE_TYPE_KEY = "ROMAN.META.EXPOSURE.TYPE"
    OPTICAL_ELEMENT_KEY = "ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT"
    SUPPORTED_PARKEYS = frozenset(
        {DETECTOR_KEY, START_TIME_KEY, EXPOSURE_TYPE_KEY, OPTICAL_ELEMENT_KEY}
    )

    # Class-level caches shared by all instances: one sync per context,
    # one bulk download per (context, reftype, exposure_type, optical_element).

    _synced_contexts = set()
    _reference_files = {}  # {(crds_context, reftype, exposure_type, optical_element): {"WFI01": "/path/file.asdf", ...}}

    def __init__(self, crds_context=None, exposure_type=WFI_TYPE_IMAGE):
        """
        Parameters
        ----------
        crds_context: str or None, optional
            CRDS context, e.g. 'roman_0042.pmap'. None uses the latest
            operational Roman context.
        exposure_type: str, optional
            Exposure type used for rmaps that select on it (e.g. DARK,
            READNOISE). One of WFI_TYPES; default WFI_TYPE_IMAGE.
        """
        if exposure_type not in WFI_TYPES:
            raise ValueError(
                f"exposure_type must be one of {sorted(WFI_TYPES)}. Got {exposure_type}."
            )
        self.crds_context = self._resolve_context(crds_context)
        self.exposure_type = exposure_type
        logging.info(
            f"CrdsHandler using CRDS context {self.crds_context}, exposure type {self.exposure_type}"
        )

    def get_reference_files(self, ref_type, optical_element=None, refresh=False):
        """
        Download the CRDS `ref_type` file that applies now for each detector and
        return a {detector: local_path} dictionary.
 
        The first call for a (context, ref_type) syncs the context, asks CRDS
        which file applies to each detector, checks which of those are already
        in CRDS_PATH, downloads only the missing ones in a single request, and
        caches the result for the rest of the process.
 
        Parameters
        ----------
        ref_type: str
            RFP ref type constant (e.g. REF_TYPE_PIXELAREA) or CRDS reftype
            name (e.g. 'gain', 'area', 'mask'). See CRDS_REF_TYPE_MAP.
        optical_element: str or None, optional
            Optical element, e.g. 'F062'. Required for reftypes whose rmap
            selects on optical element (e.g. FLAT); ignored with a warning
            for reftypes whose rmap does not.
        refresh: bool, optional
            If True, ignore the in-process cache and re-check CRDS.
 
        Returns
        -------
        dict
            {detector_name: local_file_path}, e.g.
            {'WFI01': '/crds_cache/references/roman/wfi/roman_wfi_gain_0012.asdf', ...}
        """
        reftype = self._to_crds_reftype(ref_type)
        if optical_element is not None:
            optical_element = optical_element.upper()
        key = (self.crds_context, reftype, self.exposure_type, optical_element)
 
 
        if not refresh and key in CrdsHandler._reference_files:
            return CrdsHandler._reference_files[key]
 
        # 1. Sync, then let CRDS pick one file per detector for the current time
        selected = self._select_reference_files(reftype, optical_element)

        # 2. Which of the selected files are already downloaded?
        local_paths, missing = {}, []
        for name in sorted(set(selected.values())):
            path = crds_config.locate_file(name, self.OBSERVATORY)
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
            local_paths.update(api.dump_references(self.crds_context, baserefs=missing))
 
        files = {det: local_paths[name] for det, name in sorted(selected.items())}
 
        if len(files) != len(WFI_DETECTORS):
            logging.warning(
                f"CRDS context {self.crds_context} has {reftype} files for {len(files)} "
                f"of {len(WFI_DETECTORS)} detectors: {sorted(files)}"
            )
 
        CrdsHandler._reference_files[key] = files
        return files
 

    def get_reference_file(self, ref_type, detector, optical_element=None):
        """
        Local path of the CRDS `ref_type` file for one detector (e.g. 'WFI01').
        See `get_reference_files` for optical_element.

        Raises
        ------
        KeyError
            If CRDS has no applicable file for the detector.
        """
        files = self.get_reference_files(ref_type, optical_element=optical_element)
        if detector not in files:
            raise KeyError(
                f"No CRDS {ref_type} file for detector {detector}. Available: {sorted(files)}"
            )
        return files[detector]

 
    def get_bad_pixel_map(self, detector):
        """
        Return the DQ array and local path of the CRDS MASK file for a detector.
 
        Parameters
        ----------
        detector: str
            WFI detector name, e.g. 'WFI07'.
 
        Returns
        -------
        bad_pixel_map: numpy.ndarray
            uint32 DQ array, shape (4096, 4096).
        path: str
            Local path of the CRDS MASK file.
        """
        path = self.get_reference_file(REF_TYPE_MASK, detector)
        with rdm.open(path) as ref:
            # np.array copies, so the data outlives the closed file
            bad_pixel_map = np.array(ref.dq, dtype=np.uint32)
        return bad_pixel_map, path
 
    @classmethod
    def _resolve_context(cls, crds_context=None):
        """Return crds_context, or the latest operational Roman context if None."""
        if crds_context is None:
            crds_context = crds.get_default_context(observatory=cls.OBSERVATORY, state="latest")
        return crds_context
 
    @staticmethod
    def _to_crds_reftype(ref_type):
        """
        Map an RFP ref type (e.g. REF_TYPE_PIXELAREA) to its CRDS reftype
        (e.g. 'area'). Strings not in CRDS_REF_TYPE_MAP are assumed to
        already be CRDS reftypes and are lower-cased.
        """
        return CRDS_REF_TYPE_MAP.get(ref_type.upper(), ref_type.lower())

 
    def _sync_context(self):
        """
        Sync CRDS mappings for self.crds_context, once per process.
 
        Uses `crds sync --contexts <ctx>` rather than `--all`, which pulls
        mappings for every historical context.
        """
        if self.crds_context in CrdsHandler._synced_contexts:
            return
 
        logging.info(f"Syncing CRDS mappings for context {self.crds_context}")
        try:
            result = subprocess.run(
                ["crds", "sync", "--contexts", self.crds_context],
                check=True,
                capture_output=True,
                text=True,
            )
            logging.debug(result.stdout)
            CrdsHandler._synced_contexts.add(self.crds_context)
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            stderr = getattr(err, "stderr", None) or str(err)
            logging.warning(
                f"crds sync failed for {self.crds_context}: {stderr.strip()} "
                "Continuing with the local CRDS cache / server."
            )
 
    def _select_reference_files(self, reftype, optical_element=None):
        """
        Ask CRDS which file of `reftype` applies to each detector right now.
 
        Uses the rmap's own selection rules, so for every detector this returns
        the file with the latest USEAFTER at or before the current time. A file
        whose USEAFTER is in the future is not selected. Nothing is downloaded.
 
        For rmaps that also select on exposure type (e.g. DARK, READNOISE),
        ROMAN.META.EXPOSURE.TYPE is set to self.exposure_type (WFI_IMAGE by
        default). For rmaps that select on optical element (e.g. FLAT),
        ROMAN.META.INSTRUMENT.OPTICAL_ELEMENT is set to optical_element.

        Parameters
        ----------
        reftype: str
            CRDS reftype, e.g. 'mask', 'gain', 'area'.
        optical_element: str or None, optional
            Optical element, e.g. 'F062'.
            
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
            start time, exposure type and optical element, or selects on
            optical element and none was given.
        """
        self._sync_context()
 
        rmap = (
            crds.rmap.load_mapping(self.crds_context)
            .get_imap(self.INSTRUMENT)
            .get_rmap(reftype)
        )

 
        required = set(rmap.get_required_parkeys())
        unsupported = required - self.SUPPORTED_PARKEYS
        if unsupported:
            raise ValueError(
                f"The {reftype} rmap also selects on {sorted(unsupported)}; CrdsHandler "
                "only supports detector, start time, exposure type and optical element."
            )
 
        selects_on_optical_element = self.OPTICAL_ELEMENT_KEY in required
        if selects_on_optical_element and optical_element is None:
            raise ValueError(
                f"The {reftype} rmap selects on optical element; pass optical_element, "
                "e.g. optical_element='F062'."
            )
        if optical_element is not None and not selects_on_optical_element:
            logging.warning(
                f"The {reftype} rmap does not select on optical element; "
                f"ignoring optical_element={optical_element}."
            )

 
        # CRDS standard date form, e.g. '2026-10-21 16:12:00'
        now = Time.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Selecting CRDS {reftype} files in {self.crds_context} for {now} UTC")
 
        selected = {}
        for detector in sorted(WFI_DETECTORS):
            header = {
                self.DETECTOR_KEY: detector,
                self.START_TIME_KEY: now,
                self.EXPOSURE_TYPE_KEY: self.exposure_type,
            }
            if selects_on_optical_element:
                header[self.OPTICAL_ELEMENT_KEY] = optical_element
            # get_best_ref returns a basename, None, or a "NOT FOUND ..." string
            bestref = rmap.get_best_ref(header)
            if not bestref or bestref.startswith("NOT FOUND"):
                element = f" {optical_element}" if selects_on_optical_element else ""
                logging.warning(
                    f"No CRDS {reftype}{element} file for {detector} in {self.crds_context}: {bestref}"
                    )
                continue
            selected[detector] = bestref
 
        return selected
    