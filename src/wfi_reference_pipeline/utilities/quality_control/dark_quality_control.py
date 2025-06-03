import logging

import numpy as np

from wfi_reference_pipeline.constants import (
    QC_CHECK_FAIL,
    QC_CHECK_INCOMPLETE,
    QC_CHECK_SUCCEED,
    REF_TYPE_DARK,
)

from .quality_control import QualityControl


class DarkQualityControl(QualityControl):
    def __init__(self, detector, pre_pipeline_file_list=None):
        """
        Dark Quality Control (QC) class will be attached to a Reference File Pipeline (RFP) dark object
        and import QC Reference Values from a QC yaml file. Various statistics or metrics on
        the data or data quality flags determined by the RFP are compared to the reference values that
        were written to the configuration file from the Roman Telescope Branch Data Base (RTBDB).

        IMPORTANT: Some attributes in this class are assigned directly from the quality_control_config.yml schema and must be accessed accordingly.

        pre_pipeline_file_list: list of strings or paths
            All files used for the prep_pipeline stage that must be tracked.
        """
        super().__init__(REF_TYPE_DARK, detector, pre_pipeline_file_list=pre_pipeline_file_list)


    def check_prep_pipeline(self):
        """
        Method to do checks only set to true in the config file and populate a dictionary with the checks that
        will be performed as the key and the value for that is true or false
        """
        passed = True
        for prep_method, do_check in vars(self.prep_pipeline.checks):
            try:
                if do_check and self.status_check_prep_pipeline[prep_method] == QC_CHECK_FAIL:
                    passed = False
            except KeyError:
                # This should never be hit as we imlpement a schema check
                raise KeyError(f"{prep_method} is not valid check.  Assess validity of quality control config file")
        return passed


    def check_pipeline(self):
        """
        Method to do checks only set to true in the config file and populate a dictionary with the checks that
        will be performed as the key and the value for that is true or false
        """
        for qc_method, do_check in vars(self.pipeline.checks):
            check_method = f"verify_{qc_method}"
            method = getattr(self, check_method, None)
            if callable(method):
                if do_check:
                    self.check_pipeline_results[qc_method] = method()
            else:
                # This should never be hit as we imlpement a scheme check
                raise ValueError(f"{qc_method} is not valid check.  Assess validity of quality control config file")



    def check_mean_dark_rate(self):
        """
        If this function or method is called by the flag set to true then perform this quality control check.

        Get the statistic or property from the RFP ref_type object and compare to the reference value for that check.
        Update the empty dictionary that has each
        """
        print("Executing check_mean_dark_rate")
        return
        rfp_dark_mean_rate = np.mean(self.rfp_dark.ref_type_data)  # Assuming rfp_dark_data is a numpy array
        logging.info(
            f"Mean dark rate for detector {self.rfp_dark.meta_data['detector']} mode {self.rfp_dark.meta_data['mode']} is {rfp_dark_mean_rate:.3f} dn/s")

        ref_value = self.dark_qc_reference_dict['max_mean_dark_rate_reference_value']
        logging.info(f"Compared to reference value {ref_value} for detector {self.rfp_dark_meta['detector']}")

        if rfp_dark_mean_rate < ref_value:
            return QC_CHECK_SUCCEED
        else:
            return QC_CHECK_FAIL



    def check_med_dark_rate(self):
        print("Executing check_med_dark_rate")
        return QC_CHECK_SUCCEED

    def check_std_dark_rate(self):
        print("Executing check_std_dark_rate")
        return QC_CHECK_SUCCEED

    def check_num_hot_pix(self):
        print("Executing check_num_hot_pix")
        return QC_CHECK_SUCCEED

    def check_num_dead_pix(self):
        print("Executing check_num_dead_pix")
        return QC_CHECK_SUCCEED

    def check_num_unreliable_pix(self):
        print("Executing check_num_unreliable_pix")
        return QC_CHECK_SUCCEED

    def check_num_warm_pix(self):
        print("Executing check_num_warm_pix")
        return QC_CHECK_SUCCEED






    def qc_checks_notifications(self):
        """
        There are four potential results for each test.
        QC_CHECK_FAIL, QC_CHECK_SUCCEED, QC_CHECK_WARNING, QC_CHECK_INCOMPLETE

        Any Checks skipped in the schema may still have a QC_CHECK_INCOMPLETE.  This does not affect the delivery status in any way and can be interpreted as
        QC_CHECK_SUCCEED for the below description.  If the check is NOT skipped and has QC_CHECK_INCOMPLETE, it will be interpreted as QC_CHECK_FAIL.

        If all checks in passed with QC_CHECK_SUCCEED, then ok to deliver in automated pipeline
        We should get a logging statement confirming everything and a slack notification that all is good


        If any check fails with QC_CHECK_INCOMPLETE or QC_CHECK_FAIL, then halt or pause delivery
        We should get error statements and logging statements saying that something failed with specific details on failure.
        Failure should include additional information like which test, detector, and values failed.
        In addition, an email to RFP points of contact and a slack notification

        # TODO - We Want to implement QC_CHECK_WARNING standards.  Where will will only halt a delivery if a certain amount of
        QC_WARNINGS are found.  Warnings will be made if values are with in a certain tolerance of a limit.

        """


        # qc_checks_all = True  # TODO not used
        return True

    def send_results_to_rtdb(self):
        """
        Need method to write calculated values for the quality control checks to the RTBDB RFP tables

        """

    def update_reference_table_in_rtbdb(self):
        """
        Need method to update reference values in rtbdb at some point if desired
        :return:
        """


