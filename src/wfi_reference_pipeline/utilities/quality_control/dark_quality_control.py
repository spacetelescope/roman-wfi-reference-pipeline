import logging

import numpy as np

from wfi_reference_pipeline.constants import (
    QC_CHECK_FAIL,
    QC_CHECK_SUCCEED,
    REF_TYPE_DARK,
)

from .quality_control import QualityControl


class DarkQualityControl(QualityControl):
    def __init__(self, detector, pre_pipeline_file_list=None):
        """
        Dark Quality Control (QC) class will be attached to a Reference File Pipeline (RFP) dark object
        and import QC Reference Values from a QC config file. Various statistics or metrics on
        the data or data quality flags determined by the RFP are compared to the reference values that
        were written to the configuration file from the Roman Telescope Branch Data Base (RTBDB).

        IMPORTANT: Some attributes in this class are assigned directly from the quality_control_config_<DETECTOR>.yml schema and must be accessed accordingly.

        detector: detector string unique to this pipeline run
        pre_pipeline_file_list: list of strings or paths
            All files used for the prep_pipeline stage that must be tracked.
        """
        super().__init__(
            REF_TYPE_DARK, detector, pre_pipeline_file_list=pre_pipeline_file_list
        )

    def check_prep_pipeline(self): # TODO - write these status updates to DB when implemented
        """
        Method to do checks only set to true in the config file and populate a dictionary with the checks that
        will be performed as the key and the value for that is true or false
        """
        # SAPP TODO - Evaluate once more get created if this should be moved to the QaulityContorl Base class
        # For now assume each ref type check will be unique although it doesn't look like it.
        failed_checks = []
        all_checks_passed = True
        for prep_method, do_check in vars(self.prep_pipeline.checks).items():
            for file in self.pre_pipeline_file_stems:
                try:
                    if (
                        do_check
                        and self.status_check_prep_pipeline[file][prep_method] != QC_CHECK_SUCCEED
                    ):
                        all_checks_passed = False
                        failed_checks.append(prep_method)
                except KeyError:
                    # This should never be hit as we imlpement a schema check
                    raise KeyError(
                        f"{prep_method} is not valid check.  Assess validity of quality control config file"
                    )
        if not all_checks_passed:
            failed_checks = set(failed_checks)
            raise ValueError(
                f"The following QC checks failed in prep_pipeline: {failed_checks}"
            )


    def check_pipeline(self, rfp_dark): # TODO - add results to DB when implemented
        """
        Method to do checks only set to true in the config file and populate a dictionary with the checks that
        will be performed as the key and the value for that is true or false
        """
        self.rfp_dark = rfp_dark
        self.check_pipeline_results={}
        for qc_method, do_check in vars(self.pipeline.checks).items():
            check_method = f"_check_{qc_method}"
            method = getattr(self, check_method, None)
            if callable(method):
                if do_check:
                    self.check_pipeline_results[qc_method] = method()
            else:
                # This should never be hit as we imlpement a scheme check
                raise ValueError(
                    f"{qc_method} is not valid check.  Assess validity of quality control config file"
                )

    def _check_mean_dark_rate(self):
        """
            Check whether the mean of the dark rate image is within the allowed threshold.

            This method calculates the mean of the dark rate image from the
            reference pixel dark data (`rfp_dark.dark_rate_image`) and compares it to the
            maximum allowed value defined in the pipeline configuration (`max_mean_dark_rate`).
            The result of the check is logged and returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the mean is less than or equal to the
                configured threshold; QC_CHECK_FAIL otherwise.
        """

        rfp_dark_mean_rate = np.mean(
            self.rfp_dark.dark_rate_image
        )
        ref_value = self.pipeline.values["max_mean_dark_rate"]

        if rfp_dark_mean_rate <= ref_value:
            logging.info(f"Check Mean Dark Rate passed for {self.detector}: {rfp_dark_mean_rate:.3f} <= {ref_value}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check Mean Dark Rate failed for {self.detector}: {rfp_dark_mean_rate:.3f} <= {ref_value}")
            return QC_CHECK_FAIL

    def _check_med_dark_rate(self):
        """
            Check whether the median of the dark rate image is within the allowed threshold.

            This method calculates the median of the dark rate image from the
            reference pixel dark data (`rfp_dark.dark_rate_image`) and compares it to the
            maximum allowed value defined in the pipeline configuration (`max_med_dark_rate`).
            The result of the check is logged and returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the median is less than or equal to the
                configured threshold; QC_CHECK_FAIL otherwise.
        """

        rfp_dark_median_rate = np.median(
            self.rfp_dark.dark_rate_image
        )
        ref_value = self.pipeline.values["max_med_dark_rate"]

        if rfp_dark_median_rate <= ref_value:
            logging.info(f"Check Median Dark Rate passed for {self.detector}: {rfp_dark_median_rate:.3f} <= {ref_value}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check Median Dark Rate failed for {self.detector}: {rfp_dark_median_rate:.3f} <= {ref_value}")
            return QC_CHECK_FAIL

    def _check_std_dark_rate(self):
        """
            Check whether the standard deviation of the dark rate image is within the allowed threshold.

            This method calculates the standard deviation of the dark rate image from the
            reference pixel dark data (`rfp_dark.dark_rate_image`) and compares it to the
            maximum allowed value defined in the pipeline configuration (`max_std_dark_rate`).
            The result of the check is logged and returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the standard deviation is less than or equal to the
                configured threshold; QC_CHECK_FAIL otherwise.
        """


        rfp_dark_std_rate = np.std(
            self.rfp_dark.dark_rate_image
        )
        ref_value = self.pipeline.values["max_std_dark_rate"]

        if rfp_dark_std_rate <= ref_value:
            logging.info(f"Check STD Dark Rate passed for {self.detector}: {rfp_dark_std_rate:.3f} <= {ref_value}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check STD Dark Rate failed for {self.detector}: {rfp_dark_std_rate:.3f} <= {ref_value}")
            return QC_CHECK_FAIL

    def _check_num_hot_pix(self):
        """
            Check whether the number of hot pixels exceeds the allowed threshold.

            This method counts the number of pixels in the data quality (DQ) mask
            that are flagged as hot using the `HOT` bit flag. It then compares
            this count against the maximum allowable number of hot pixels defined
            in the pipeline configuration. The result of the check is logged and
            returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the number of hot pixels is within the allowed
                limit; QC_CHECK_FAIL otherwise.
        """
        hot_pixel_count = np.count_nonzero(self.rfp_dark.dq_mask & self.rfp_dark.dqflag_defs["HOT"])
        max_num_hot_pix = self.pipeline.values["max_num_hot_pix"]

        if hot_pixel_count <= max_num_hot_pix:
            logging.info(f"Check Max Num Hot Pixels Dark Rate passed for {self.detector}: {hot_pixel_count} <= {max_num_hot_pix}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check Max Num Hot Pixels Dark Rate failed for {self.detector}: {hot_pixel_count} <= {max_num_hot_pix}")
            return QC_CHECK_FAIL


    def _check_num_dead_pix(self):
        """
            Check whether the number of dead pixels exceeds the allowed threshold.

            This method counts the number of pixels in the data quality (DQ) mask
            that are flagged as dead using the `DEAD` bit flag. It then compares
            this count against the maximum allowable number of dead pixels defined
            in the pipeline configuration. The result of the check is logged and
            returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the number of dead pixels is within the allowed
                limit; QC_CHECK_FAIL otherwise.
        """
        dead_pixel_count = np.count_nonzero(self.rfp_dark.dq_mask & self.rfp_dark.dqflag_defs["DEAD"])
        max_num_dead_pix = self.pipeline.values["max_num_dead_pix"]
        if dead_pixel_count <= max_num_dead_pix:
            logging.info(f"Check Max Num Dead Pixels Dark Rate passed for {self.detector}: {dead_pixel_count} <= {max_num_dead_pix}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check Max Num Dead Pixels Dark Rate failed for {self.detector}: {dead_pixel_count} <= {max_num_dead_pix}")
            return QC_CHECK_FAIL

    def _check_num_warm_pix(self):
        """
            Check whether the number of warm pixels exceeds the allowed threshold.

            This method counts the number of pixels in the data quality (DQ) mask
            that are flagged as warm using the `WARM` bit flag. It then compares
            this count against the maximum allowable number of warm pixels defined
            in the pipeline configuration. The result of the check is logged and
            returned as a quality control (QC) status.

            Returns
            -------
            int
                QC_CHECK_SUCCEED if the number of warm pixels is within the allowed
                limit; QC_CHECK_FAIL otherwise.
        """
        warm_pixel_count = np.count_nonzero(self.rfp_dark.dq_mask & self.rfp_dark.dqflag_defs["WARM"])
        max_num_warm_pix = self.pipeline.values["max_num_warm_pix"]
        if warm_pixel_count <= max_num_warm_pix:
            logging.info(f"Check Max Num Warm Pixels Dark Rate passed for {self.detector}: {warm_pixel_count} <= {max_num_warm_pix}")
            return QC_CHECK_SUCCEED
        else:
            logging.warning(f"Check Max Num Warm Pixels Dark Rate failed for {self.detector}: {warm_pixel_count} <= {max_num_warm_pix}")
            return QC_CHECK_FAIL

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

    def _update_reference_table_in_rtbdb(self):
        """
        Need method to update reference values in rtbdb at some point if desired
        :return:
        """
