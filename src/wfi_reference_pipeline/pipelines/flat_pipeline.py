import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from romancal.refpix import RefPixStep
from romancal.linearity import LinearityStep
from romancal.dark_current import DarkCurrentStep
from romancal.ramp_fitting import RampFitStep


from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.constants import REF_TYPE_FLAT
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.filename_parser import FilenameParser
from wfi_reference_pipeline.utilities.logging_functions import log_info


class FlatPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Dark Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save outputs to go into superdark
    prep_superdark_file: Prepares the superdark file input to be used as input for run_pipeline
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    flat_pipeline = FlatPipeline()
    flat_pipeline.select_uncal_files()
    flat_pipeline.prep_pipeline()
    flat_pipeline.run_pipeline()

    or

    flat_pipeline.restart_pipeline()

    """

    def __init__(self):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FLAT)
        self.flat_file = None

    @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("FLAT SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        # files = [str(file) for file in self.ingest_path.glob("r0044401001001001001_01101_000*_WFI01_uncal.asdf")]
        files = list(
            # self.ingest_path.glob("r0044401001001001001_01101_0001_WFI01_uncal.asdf")
            self.ingest_path.glob("r00444*_WFI01_uncal.asdf")
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
            result = DQInitStep.call(in_file, save_results=False)
            result = RefPixStep.call(result, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = LinearityStep.call(result, save_results=False)
            result = DarkCurrentStep.call(result, save_results=False)
            result = RampFitStep.call(result, save_results=False)
            # TODO Need to confirm steps from romancal and their functionality

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

        logging.info(
            "Finished PREPPING files to make FLAT reference file from RFP")

        logging.info("Starting to make FLAT from PREPPED FLAT asdf files")
