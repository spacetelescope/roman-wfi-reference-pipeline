import sys
from abc import ABC, abstractmethod
from pathlib import Path

from constants import REF_TYPE_DARK, WFI_DETECTORS

from wfi_reference_pipeline.config.config_access import get_data_files_config
from wfi_reference_pipeline.utilities.file_handler import FileHandler
from wfi_reference_pipeline.utilities.logging_functions import configure_logging
from wfi_reference_pipeline.utilities.quality_control.dark_quality_control import (
    DarkQualityControl,
)


class Pipeline(ABC):
    """
    Base Class to be used with all reference type pipeline derived classes:

    Enforces template of:
    Standard reftype agnostic initialization
    Automatic logging configuration for derived class
    abstractmethod - Selecting level 1 uncalibrated asdf files
    abstractmethod - Preparing the pipeline using romancal routines
    abstractmethod - Running the pipeline to calibrate the data in the reference type specific pipeline
    abstractmethod - Running any actions/checks needed before delivering final product to CRDS
    abstractmethod - Deliver final product and do any post delivery work
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

    @abstractmethod
    def pre_deliver(self):
        """
        Perform all tasks and checks before delivering the final product
        """
        pass

    @abstractmethod
    def deliver(self):
        """
        Deliver the final product to the end user.
        Perform any actions dependent upon delivery.
        """
        pass

    def init_quality_control(self):
        if self.ref_type == REF_TYPE_DARK:
            self.qc = DarkQualityControl(self.detector, pre_pipeline_file_list=self.uncal_files)
        # elif self.ref_type == REF_TYPE_FLAT:
        # elif self.ref_type == REF_TYPE_MASK:
        # elif self.ref_type == REF_TYPE_READNOISE:
        # elif self.ref_type == REF_TYPE_REFPIX:
        else:
            raise ValueError(f"Reference Type {self.ref_type} does not yet have quality control established")

    def restart_pipeline(self):
        """
        Run all steps of the pipeline.

        Note: if updating, remember to also update DarkPipeline.restart_pipeline
        """
        self.select_uncal_files()
        self.init_quality_control()
        self.prep_pipeline()
        self.run_pipeline()
        self.pre_deliver()
        self.deliver()

