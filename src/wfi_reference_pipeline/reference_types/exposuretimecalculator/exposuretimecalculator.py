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

    def load_detector_config(self):
        """Read YAML and merge static + detector-specific settings."""
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        static_cfg = cfg.get("static", {})
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
            self.load_detector_config()

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


ETC_CONFIG = (Path(__file__).parent.parent.parent / "config" / "exposure_time_calculator_config.yml").resolve()


def update_etc_config_from_crds(etc_dump_dir="/grp/roman/RFP/DEV/scratch/etc_dump_files/"):
    """
    Load the ETC YAML configuration and set CRDS server & cache path.

    Parameters
    ----------
    etc_dump_dir : str
        Path to the directory where CRDS reference files will be cached/downloaded.

    crds_server = "https://roman-crds.stsci.edu/"
    crds_context = crds.get_default_context()  # or specify a context if desired

    print(f"CRDS server: {crds_server}")
    print(f"CRDS context: {crds_context}")
    os.environ["CRDS_PATH"] = etc_dump_dir
    os.environ["CRDS_SERVER_URL"] = crds_server

    print(f"CRDS_PATH set to: {os.environ['CRDS_PATH']}")
    print(f"CRDS_SERVER_URL set to: {os.environ['CRDS_SERVER_URL']}")
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

    # pmap = crds.rmap.asmapping(crds_context)
    # imap = pmap.get_imap("wfi")
    # readnoise_rmap = imap.get_rmap("readnoise")
    # all_readnoise_files = readnoise_rmap.reference_names()

    readnoise_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('readnoise').reference_names()
    print(readnoise_files)
    results = api.dump_references(crds_context, readnoise_files)



    #return cfg

    # for det_id, det_block in cfg.get("detectors", {}).items():
    #     # Need to make sure file lists are returned or sorted in order of detector ID WFI018-WFI18
    #     dark_file_list = query_crds_files(detector=det_id, reftype='dark')
    #     read_file_list = query_crds_files(detector=det_id, reftype='readnoise')
    #     flat_file_list = query_crds_files(detector=det_id, reftype='flat')
    #     sat_file_list = query_crds_files(detector=det_id, reftype='saturation')

    #     # Compute statistics
    #     new_dark = cls.mean_data_from_asdf_files(dark_file_list)
    #     new_read = cls.mean_data_from_asdf_files(read_file_list)
    #     new_ff_e = cls.stddev_data_from_asdf_files(flat_file_list)
    #     new_fw = cls.median_data_from_asdf_files(sat_file_list)

    #     det_block.update({
    #         "dark_current_avg": new_dark,
    #         "readnoise_avg": new_read,
    #         "saturation_fullwell": new_fw,
    #         "ff_electrons": new_ff_e
    #     })

    #     with open(config_path, "w") as f:
    #         yaml.safe_dump(cfg, f, sort_keys=False)

    #     print(f"Configuration file '{config_path}' updated from CRDS.")




# ---------------- Helper functions ----------------
@staticmethod
def compute_stat_from_asdf_files(file_list, stat_func):
    """
    Open each ASDF file in file_list and compute a statistic using stat_func (e.g., np.mean, np.median, np.std).
    Raises an exception if any file cannot be opened.
    """
    vals = []
    for fname in file_list:
        try:
            with asdf.open(fname) as af:
                data = af.tree['roman']['data']
                vals.append(stat_func(data))
        except Exception as e:
            raise RuntimeError(f"Error opening ASDF file '{fname}': {e}") from e

    if not vals:
        raise ValueError("No valid data found in any of the provided ASDF files.")
    return vals

# These are thin wrappers for clarity
mean_data_from_asdf_files = staticmethod(lambda files: ExposureTimeCalculator.compute_stat_from_asdf_files(files, np.mean))
median_data_from_asdf_files = staticmethod(lambda files: ExposureTimeCalculator.compute_stat_from_asdf_files(files, np.median))
stddev_data_from_asdf_files = staticmethod(lambda files: ExposureTimeCalculator.compute_stat_from_asdf_files(files, np.std))