import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
import roman_datamodels as rdm
import os
import yaml
import crds
from crds.client import api
import subprocess
import shutil
from pathlib import Path

from ..reference_type import ReferenceType


class ExposureTimeCalculator(ReferenceType):
    """
    Class ExposureTimeCalculator() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_etc_file() creates or retrieves the exposure time calculator necessary
    data from an input file.
    """

    def __init__(self,
                 meta_data: dict,
                 file_list=None,
                 outfile="roman_etc_file.asdf",
                 clobber=False,
                 config_path="exposure_time_calculator_config.yml"):
        """
        Parameters
        ----------
        meta_data : dict
            Must include a key like {"detector": "WFI03"} to identify the detector.
        file_list : list[str] | None
            When creating a reference file, this should contain the YAML config path.
        outfile : str
            Output ASDF file name for Mode 1.
        clobber : bool
            Whether to overwrite existing ASDF file.
        config_path : str
            Path to the YAML configuration file.
        """
        super().__init__(meta_data, file_list, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaETC):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaETC"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI ETC reference file."

        self.outfile = outfile
        self.config_path = Path(config_path)
        self.detector_id = self.meta.get("detector")   # e.g., "WFI07"
        self.detector_config = None

    def get_detector_config(self):
        """Read YAML and merge static + detector-specific settings."""
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        static_cfg = cfg.get("common", {})
        det_cfgs = cfg.get("detectors", {})

        if self.detector_id not in det_cfgs:
            raise ValueError(f"Detector {self.detector_id} not in config file")

        self.detector_config = {**static_cfg, **det_cfgs[self.detector_id]}
        return self.detector_config

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the exposure time calculator
        using the merged detector configuration.
        """
        if self.detector_config is None:
            self.get_detector_config()

        # Example datamodel structure; replace rds.ExposureTimeRef
        # with the correct roman_datamodels reference class when available.
        etc_datamodel_tree = rds.ExposureTimeRef()
        etc_datamodel_tree["meta"] = self.meta
        etc_datamodel_tree["data"] = self.detector_config
        return etc_datamodel_tree
    

    def save_exposure_time_file(self, datamodel_tree=None):
        """Write the ASDF file for the detector."""
        af = asdf.AsdfFile()
        af.tree = {"roman": datamodel_tree or self.populate_datamodel_tree()}
        af.write_to(self.outfile, overwrite=True)



# -------------------------------
# Standalone function to update config file
# -------------------------------

ETC_CONFIG = (Path(__file__).parent.parent.parent / "config" / "exposure_time_calculator_config.yml").resolve()

def update_etc_config_from_crds(etc_dump_dir="/grp/roman/RFP/DEV/scratch/etc_dump_files/"):
    """
    Update ETC YAML config with median readnoise, dark current, and flat field
    values for each WFI detector from CRDS reference files.
    """
    """
    Load the ETC YAML configuration and set CRDS server & cache path.

    Parameters
    ----------
    etc_dump_dir : str
        Path to the directory where CRDS reference files will be cached/downloaded.
    """

    print("CRDS_SERVER_URL:", os.environ.get("CRDS_SERVER_URL"))
    print("CRDS_PATH:", os.environ.get("CRDS_PATH"))
    crds_context = crds.get_default_context()
    print(f"CRDS context: {crds_context}")

    if os.path.exists(etc_dump_dir):
        print(f"Deleting existing dump directory: {etc_dump_dir}")
        shutil.rmtree(etc_dump_dir)

    print(f"Creating dump directory: {etc_dump_dir}")
    os.makedirs(etc_dump_dir, exist_ok=True)

    print("Syncing CRDS reference files...")
    try:
        result = subprocess.run(
            ["crds", "sync", "--all"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running crds sync: {e.stderr}")


    if not os.path.exists(ETC_CONFIG):
        raise FileNotFoundError(f"Config file not found at {ETC_CONFIG}")

    with open(ETC_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    # get a pointer to just the detectors portion of the ETC CONFIG file
    detectors_cfg = cfg.get("detectors", {})

    # -------------------------------
    # Locally download all reference files needed to update ETC config
    # -------------------------------

    readnoise_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('readnoise').reference_names()
    results = api.dump_references(crds_context, readnoise_files)
    readnoise_filepaths = list(results.values())

    dark_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('dark').reference_names()
    results = api.dump_references(crds_context, dark_files)
    dark_filepaths = list(results.values())

    # TODO figure out a way to just download one optical element for all 18 detectors using this
    # or some modification to this code
    flat_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('flat').reference_names()
    results = api.dump_references(crds_context, flat_files)
    flat_filepaths = list(results.values()) 

    saturation_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('saturation').reference_names()
    results = api.dump_references(crds_context, saturation_files)
    saturation_filepaths = list(results.values())

    # -------------------------------
    # READNOISE: update readnoise_avg with median readnoise from data array
    # -------------------------------
    for filepath in readnoise_filepaths:
        try:
            ref = rdm.open(filepath)
            det = ref.meta.instrument.detector
            val = float(np.median(ref.data))
            if det in detectors_cfg:
                detectors_cfg[det]["readnoise_avg"] = val
                detectors_cfg[det]["readnoise_on"] = True
                print(f"{det}: readnoise_avg -> {val:.2f}")
            else:
                print(f"Warning: detector {det} not found in config.")
        except Exception as e:
            print(f"Failed to process readnoise {filepath}: {e}")

    # -------------------------------
    # DARK CURRENT: update dark_current_avg with mean of dark current rate array
    # -------------------------------
    for filepath in dark_filepaths:
        try:
            ref = rdm.open(filepath)
            det = ref.meta.instrument.detector
            val = float(np.mean(ref.dark_slope))
            if det in detectors_cfg:
                detectors_cfg[det]["dark_current_avg"] = val
                detectors_cfg[det]["dark_current_on"] = True
                print(f"{det}: dark_current_avg -> {val:.3f}")
            else:
                print(f"Warning: detector {det} not found in config.")
        except Exception as e:
            print(f"Failed to process dark current {filepath}: {e}")

    # -------------------------------
    # FLAT FIELD: update ff_electrons
    # -------------------------------
    for filepath in flat_filepaths:
        try:
            ref = rdm.open(filepath)
            if ref.meta.instrument.optical_element == 'F062':
                det = ref.meta.instrument.detector
                std_val = float(np.std(ref.data))
                ff_electrons = 1.0 / (std_val ** 2)
                if det in detectors_cfg:
                    detectors_cfg[det]["ff_electrons"] = ff_electrons
                    detectors_cfg[det]["ffnoise"] = True
                    print(f"{det}: ff_electrons -> {ff_electrons:.2f}")
                else:
                    print(f"Warning: detector {det} not found in config.")
        except Exception as e:
            print(f"Failed to process flat field {filepath}: {e}")

    # -------------------------------
    # SATURATION: update saturation_fullwell
    # -------------------------------
    for filepath in saturation_filepaths:
        try:
            ref = rdm.open(filepath)
            det = ref.meta.instrument.detector
            val = float(np.amax(ref.data))
            if det in detectors_cfg:
                detectors_cfg[det]["saturation_fullwell"] = val
                detectors_cfg[det]["saturation_on"] = True
                print(f"{det}: saturation_fullwell -> {val:.1f}")
            else:
                print(f"Warning: detector {det} not found in config.")
        except Exception as e:
            print(f"Failed to process saturation {filepath}: {e}")


    # -------------------------------
    # Write updated config
    # -------------------------------
    backup_path = ETC_CONFIG.with_suffix(".bak")
    shutil.copy(ETC_CONFIG, backup_path)
    print(f"\nBackup saved to: {backup_path}")

    with open(ETC_CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Updated config file saved to: {ETC_CONFIG}")





