from abc import ABC, abstractmethod
from pathlib import Path

from wfi_reference_pipeline.utilities.config_handler import get_datafiles_config
from wfi_reference_pipeline.utilities.logging_functions import configure_logging

class Pipeline(ABC):

    def __init__(self):
        self._uncal_files = []
        self._files_prepped = []
        self._datamodels_prepped = []
        # Initialize logging named for the derived class
        configure_logging(f"{self.__class__.__name__}")
        try:
            self._datafiles_config = get_datafiles_config()
            self.ingest_path = Path(self._datafiles_config["ingest_dir"])                   # TODO SAPP - FIGURE THIS OUT
            self.prep_path = Path(self._datafiles_config["prep_dir"])                       # TODO SAPP - FIGURE THIS OUT
            self.calibrated_out_path = Path(self._datafiles_config["calibrated_dir"])       # TODO SAPP - FIGURE THIS OUT
        except (FileNotFoundError, ValueError) as e:
            print("ERROR READING CONFIG FILE - {e}")
            exit()

    @abstractmethod
    def select_uncal_files(self):
        pass

    @abstractmethod
    def prep_pipeline(self, file_list):
        pass

    @abstractmethod
    def run_pipeline(self, file_list):
        pass

    def restart_pipeline(self):
        self.select_uncal_files()
        self.prep_pipeline(self._uncal_files)
        self.run_pipeline(self._files_prepped) # TODO change this to self.readnoise_datamodels_prepped when code can handle it.

