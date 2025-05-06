import sys
from abc import ABC, abstractmethod
from pathlib import Path

from constants import WFI_DETECTORS

from wfi_reference_pipeline.config.config_access import get_data_files_config
from wfi_reference_pipeline.utilities.file_handler import FileHandler
from wfi_reference_pipeline.utilities.logging_functions import configure_logging


class Pipeline(ABC):
    """
    Base Class to be used with all reference type pipeline derived classes:

    Enforces template of:
    Standard reftype agnostic initialization
    Automatic logging configuration for derived class
    abstractmethod - Selecting level 1 uncalibrated asdf files
    abstractmethod - Preparing the pipeline using romancal routines
    abstractmethod - Running the pipeline to calibrate the data in the reference type specific pipeline
    Restart_pipeline general functionality (run from scratch)

    """

    def __init__(self, ref_type, detector):
        self.uncal_files = []
        self.prepped_files = []
        self.ref_type = ref_type

        if detector.upper() in WFI_DETECTORS:
            self.detector = detector.upper()
        else:
            raise KeyError (f"Invalid Detector {detector} - choose from {WFI_DETECTORS}")

        try:
            # Initialize logging named for the derived class
            configure_logging(f"{self.__class__.__name__}")
            self._data_files_config = get_data_files_config()
            self.ingest_path = Path(self._data_files_config["ingest_dir"])
            self.prep_path = Path(self._data_files_config["prep_dir"])
            self.pipeline_out_path = Path(self._data_files_config["crds_ready_dir"])
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR READING CONFIG FILE - {e}")
            sys.exit()
        self.file_handler = FileHandler(self.ref_type, self.prep_path, self.pipeline_out_path)


    @abstractmethod
    def select_uncal_files(self):
        """
        Select the level 1 uncalibrated asdf files
        """
        pass

    @abstractmethod
    def prep_pipeline(self, file_list):
        """
        Preparing the pipeline using romancal routines
        """
        pass

    @abstractmethod
    def run_pipeline(self, file_list):
        """
        Run the pipeline to calibrate the data in the reference type specific pipeline
        """
        pass

    def restart_pipeline(self):
        self.select_uncal_files()
        self.prep_pipeline()
        self.run_pipeline()

