import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.refpix import refpix
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.constants import REF_TYPE_FGS_MASK
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.fgs_mask.fgs_mask import FGSMask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# from wfi_reference_pipeline.utilities.logging_functions import log_info


class FGSMaskPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all FGS Mask Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save output
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    fgs_mask_pipeline = FGSMaskPipeline("<detector string>")
    fgs_mask_pipeline.select_uncal_files()
    fgs_mask_pipeline.prep_pipeline(fgs_mask_pipeline.uncal_files)
    fgs_mask_pipeline.run_pipeline(fgs_mask_pipeline.prepped_files)
    fgs_mask_pipeline.pre_deliver()
    fgs_mask_pipeline.deliver()

    or

    fgs_mask_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FGS_MASK, detector)

    # @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("FGS_MASK SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """

        # Get files from input directory
        files = [
            str(file)
            for file in self.ingest_path.glob(
                "r0044401001001001001_01101_000*_WFI01_uncal.asdf"
            )
        ]
        # files = [str(file) for file in self.ingest_path.glob("*_WFI01_uncal.asdf")]
        # files = list(
        #     self.ingest_path.glob("r0032101001001001001_01101_0001_WFI01_uncal.asdf")
        # )

        self.uncal_files = files
        logging.info(f"Ingesting {len(files)} Files: {files}")

    # @log_info
    def prep_pipeline(self, file_list=None):
        """Prepare calibration data files by running data through select romancal steps"""
        logging.info("FGS_MASK PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = list(map(Path, self.uncal_files))

        for file in file_list:
            logging.info("OPENING - " + file.name)
            in_file = rdm.open(file)

            # If save_result = True, then the input asdf file is written to disk, in the current directory, with the
            # name of the last step replacing 'uncal'.asdf
            result = DQInitStep.call(in_file, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = refpix.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

        logging.info(
            "Finished PREPPING files to make FGS_MASK reference file from RFP"
        )

    # @log_info
    def run_pipeline(self, file_list=None):

        logging.info("FGS_MASK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.prepped_files

        tmp = MakeDevMeta(
            ref_type=self.ref_type
        )  # TODO replace with MakeMeta which gets actual information from files
        # fgs_mask_dev_meta = tmp.meta_fgs_mask.export_asdf_meta()
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_fgs_mask.mode,
            tmp.meta_fgs_mask.instrument_detector,
        )

        rfp_fgs_mask = FGSMask(
            meta_data=tmp.meta_fgs_mask,
            file_list=file_list,
            ref_type_data=None,
            outfile=out_file_path,
            clobber=True,
        )
        rfp_fgs_mask.make_fgs_mask_image()
        rfp_fgs_mask.generate_outfile()
        logging.info("Finished RFP to make FGS_MASK")
        print("Finished RFP to make FGS_MASK")
