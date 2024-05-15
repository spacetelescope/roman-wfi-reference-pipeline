import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from wfi_reference_pipeline.constants import REF_TYPE_DARK
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.reference_types.dark.dark import SuperDark
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.logging_functions import log_info
from wfi_reference_pipeline.utilities.rtbdb_functions import get_ma_table_from_rtbdb


class DarkPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Dark Pipeline functionality
    Gives user access to:
        select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
        prep_pipeline : Preparing the pipeline using romancal routines and save outputs to go into superdark
        run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
        restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
        dark_pipeline = DarkPipeline()
        dark_pipeline.select_uncal_files()
        dark_pipeline.prep_pipeline(dark_pipeline.uncal_files)
        dark_pipeline.run_pipeline(dark_pipeline.prepped_files)

        or

        dark_pipeline.restart_pipeline()

    """

    def __init__(self):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_DARK)

    @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("DARK SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        # files = [str(file) for file in self.ingest_path.glob("r0044401001001001001_01101_000*_WFI01_uncal.asdf")]
        files = list(
            self.ingest_path.glob("r0032101001001001001_01101_0001_WFI01_uncal.asdf")
        )

        self.uncal_files = files
        logging.info(f"Ingesting {len(files)} Files: {files}")

    @log_info
    def prep_pipeline(self, file_list=None):
        logging.info("DARK PREP")

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
            result = SaturationStep.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

        logging.info(
            "Finished PREPPING files to make DARK reference file from RFP"
        )

        logging.info(
            "Starting to make SUPERDARK from PREPPED DARK asdf files"
        )


    @log_info
    def run_pipeline(self, file_list=None):
        logging.info("DARK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.prepped_files

        tmp = MakeDevMeta(
            ref_type=self.ref_type
        )
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_dark.mode,
            tmp.meta_dark.instrument_detector,
        )

        rfp_dark = Dark(meta_data=tmp.meta_dark,
                        file_list=file_list,
                        data_array=None,
                        outfile=out_file_path,
                        clobber=True
        )
        ma_table_dict = get_ma_table_from_rtbdb(ma_table_number=3)
        rfp_dark.make_ma_table_resampled_cube(read_pattern=ma_table_dict[read_pattern])
        rfp_dark.make_dark_rate_image()
        rfp_dark.generate_outfile()
        logging.info("Finished RFP to make DARK")
        print("Finished RFP to make DARK")
