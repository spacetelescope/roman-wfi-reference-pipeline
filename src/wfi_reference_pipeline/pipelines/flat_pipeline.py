import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dark_current import DarkCurrentStep
from romancal.dq_init import DQInitStep
from romancal.linearity import LinearityStep
from romancal.ramp_fitting import RampFitStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.constants import REF_TYPE_FLAT
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.logging_functions import log_info


class FlatPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Dark Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save outputs to go into superdark
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    flat_pipeline = FlatPipeline("<detector string>")
    flat_pipeline.select_uncal_files()
    flat_pipeline.prep_pipeline()
    flat_pipeline.run_pipeline()
    flat_pipeline.pre_deliver()
    flat_pipeline.deliver()

    or

    flat_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FLAT, detector)
        self.flat_file = None

    @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("FLAT SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        files = list(
            self.ingest_path.glob("*_uncal.asdf")  # Change this
        )

        self.uncal_files = files
        logging.info(f"Ingesting {len(files)} Files: {files}")

    @log_info
    def prep_pipeline(self, file_list=None):
        logging.info("FLAT PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.uncal_files

        for file in file_list:
            logging.info("OPENING - " + file.name)
            in_file = rdm.open(file)

            # If save_result = True, then the input asdf file is written to disk, in the current directory, with the
            # name of the last step replacing 'uncal'.asdf
            if in_file["meta"]["cal_step"]["dark"] == "INCOMPLETE":
                result = DQInitStep.call(in_file, save_results=False)
            if in_file["meta"]["cal_step"]["refpix"] == "INCOMPLETE":
                result = RefPixStep.call(result, save_results=False)
            if in_file["meta"]["cal_step"]["saturation"] == "INCOMPLETE":
                result = SaturationStep.call(result, save_results=False)
            if in_file["meta"]["cal_step"]["linearity"] == "INCOMPLETE":
                result = LinearityStep.call(result, save_results=False)
            if in_file["meta"]["cal_step"]["dark"] == "INCOMPLETE":
                result = DarkCurrentStep.call(result, save_results=False)
            if in_file["meta"]["cal_step"]["ramp_fil"] == "INCOMPLETE"
                result = RampFitStep.call(result, save_results=False)
            # Make sure to only use files that have not been flat-fielded
            if in_file["meta"]["cal_step"]["flat_field"] == "INCOMPLETE":
                prep_output_file_path = self.file_handler.format_prep_output_file_path(
                    result.meta.filename
                )
                result.save(path=prep_output_file_path)

                self.prepped_files.append(prep_output_file_path)

        logging.info(
            "Finished PREPPING files to make FLAT reference file from RFP")

        logging.info("Starting to make FLAT from PREPPED FLAT asdf files")

    @log_info
    def run_pipeline(self, file_list=None):
        logging.info("FLAT PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            if self.prepped_files is not None:
                file_list = self.prepped_files
            else:
                raise ValueError(
                    'Prepare file or pass a (pre-processed) file list')

        tmp = MakeDevMeta(ref_type=self.ref_type)
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_flat.mode,
            tmp.meta_flat.instrument_detector,
        )

        rfp_flat = Flat(
            meta_data=tmp.meta_flat,
            file_list=file_list,
            ref_type_data=None,
            outfile=out_file_path,
            clobber=True,
        )
        rfp_flat.flat_image = rfp_flat.make_flat_from_files(file_list)
        rfp_flat.populate_datamodel_tree()
        rfp_flat.generate_outfile()
        logging.info("Finished RFP to make FLAT")
        print("Finished RFP to make FLAT")
