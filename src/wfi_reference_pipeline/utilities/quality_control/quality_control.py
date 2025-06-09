from pathlib import Path

from wfi_reference_pipeline.config.config_access import get_quality_control_config
from wfi_reference_pipeline.constants import (
    QC_CHECK_CAUTION,
    QC_CHECK_FAIL,
    QC_CHECK_INCOMPLETE,
    QC_CHECK_SUCCEED,
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
    """
    QualityControl(QC) class will be utilized as a base class for reference type specific Quality Control classes.
    Its main purpose is to hold the ref type, detector, and extract all elements of the quality control config files and assign
    them as actual attributes within the class.

    This means that every element in the quality_control_config_<DETECTOR>.yml will become an attribute of the QC object.
        For example if quality_control_config_WFI01.yml contains:
            prep_pipeline:
                checks:
                    dqinit: true
                    saturation: true
                    refpix: true
        Then our QC object will generate:
            qc.prep_pipeline.checks.dqinit == true
            qc.prep_pipeline.checks.saturation == true
            qc.prep_pipeline.checks.refpix == true

    Input Parameters:
        ref_type: reference type from constants.py
        detector: detector string unique to this pipeline run
        pre_pipeline_file_list: list of strings or paths
            All files used for the prep_pipeline stage that must be tracked.

    """

    def __init__(self, ref_type, detector, pre_pipeline_file_list=None):

        self.ref_type = ref_type
        self.detector = detector
        quality_control = get_quality_control_config(ref_type, detector)
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

        self.pre_pipeline_methods = [prep_method for prep_method, _ in vars(self.prep_pipeline.checks).items()]

        self._init_prep_pipeline()

    def _init_prep_pipeline(self):
        """
        Initialize all prep files to be marked INCOMPLETE for every method in prep_pipeline using nested dicts
        """
        self.status_check_prep_pipeline = {}

        for file in self.pre_pipeline_file_stems:
            if file not in self.status_check_prep_pipeline:
                # If we dont have this file yet, make a new dict for it
                self.status_check_prep_pipeline[file]={}
            for method_key in self.pre_pipeline_methods:
                self.update_prep_pipeline_file_status(file, method_key, QC_CHECK_INCOMPLETE)

    def _get_qc_status_from_string(self, status):
        """
        Take a status and return the corresponding qc_status for internal tracking
        IMPORTANT: All methods saved in this list must be indexed using the same naming convention as quality_control_config.yml
        """

        if type(status) is str:
            if status == "SKIPPED":
                qc_status = QC_CHECK_INCOMPLETE
            elif status == "INCOMPLETE":
                qc_status = QC_CHECK_INCOMPLETE
            elif status == "COMPLETE":
                qc_status = QC_CHECK_SUCCEED
            else:
                raise ValueError(f"QC Status {status} is invalid")
        elif type(status) is int:
            qc_status = status
        else:
            raise TypeError(f"Invalid qc status type: {type(status)}")
        return qc_status


    def update_prep_pipeline_file_status(self, file_name, method, status):
        """
        Update status list for each file and the respective prep pipeline method
        IMPORTANT: All methods saved in this list must be indexed using the same naming convention as quality_control_config.yml

        """
        # Method should be the same as the attributes in the config file
        if method in self.pre_pipeline_methods:
            file_key = Path(file_name).stem
            qc_status = self._get_qc_status_from_string(status)
            if qc_status in VALID_QC_STATUS:
                self.status_check_prep_pipeline[file_key][method] =  qc_status
            else:
                raise ValueError(f"Invalid status {qc_status}, use defined - QC_CHECK_FAIL, QC_CHECK_SUCCEED, QC_CHECK_CAUTION, QC_CHECK_INCOMPLETE")
        else:
            raise ValueError(f"Invalid method {method}, use {self.pre_pipeline_methods}")


