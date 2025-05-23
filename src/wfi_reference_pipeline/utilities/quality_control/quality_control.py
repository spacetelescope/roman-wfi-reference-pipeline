from pathlib import Path

from wfi_reference_pipeline.config.config_access import get_quality_control_config
from wfi_reference_pipeline.constants import (
    QC_CHECK_FAIL,
    QC_CHECK_SUCCEED,
    QC_CHECK_CAUTION,
    QC_CHECK_INCOMPLETE
)

VALID_QC_STATUS = [QC_CHECK_CAUTION, QC_CHECK_FAIL, QC_CHECK_INCOMPLETE, QC_CHECK_SUCCEED]

class _ObjectConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = _ObjectConfig(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

class QualityControl(_ObjectConfig):

    def __init__(self, ref_type, pre_pipeline_file_list=None):

        self.ref_type = ref_type
        quality_control = get_quality_control_config(ref_type)
        super().__init__(quality_control)

        # TODO - Consider utilizing restart marker to know if this quality control object was initiated during pipeline restart or manually
        self.restart = False
        self.pre_pipeline_file_stems = []
        if pre_pipeline_file_list:
            file_list = list(map(Path, pre_pipeline_file_list))
            for file in file_list:
                self.pre_pipeline_file_stems.append(file.stem)
        else:
            # Populate pre_pipeline_file_list from the database - TODO
            raise KeyError("Must send in pre_pipeline_file_list until database is implemented")

        self._init_prep_pipeline()

    def _init_prep_pipeline(self):
        self.status_check_prep_pipeline = {}
        #initialize all prep files to be either FAIL or SKIP for every method in prep_pipeline using nested dicts
        for file in self.pre_pipeline_file_stems:
            if file not in self.status_check_prep_pipeline:
                self.status_check_prep_pipeline[file]={}
            for prep_method, _ in vars(self.prep_pipeline.checks).items():
                self.update_prep_pipeline_file_status(file, prep_method, QC_CHECK_INCOMPLETE)

    def update_prep_pipeline_file_status(self, file, method, status):
        """
        Update status list for each file and the respective prep pipeline method

        IMPORTANT: All methods saved in this list must be indexed using the same naming convention as quality_control_config.yml
        """
        if status in VALID_QC_STATUS:
            self.status_check_prep_pipeline[file][method] =  status
        else:
            raise ValueError(f"Invalid status {status}, use defined - QC_CHECK_FAIL, QC_CHECK_SUCCEED, QC_CHECK_CAUTION, QC_CHECK_INCOMPLETE")

