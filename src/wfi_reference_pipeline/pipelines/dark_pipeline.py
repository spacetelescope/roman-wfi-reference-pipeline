import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.config.config_access import get_pipelines_config
from wfi_reference_pipeline.constants import (
    DARK_LONG_IDENTIFIER,
    DARK_LONG_NUM_READS,
    DARK_SHORT_NUM_READS,
    DARK_SIGMA_CLIP_SD_HIGH,
    DARK_SIGMA_CLIP_SD_LOW,
    REF_TYPE_DARK,
)
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.reference_types.dark.superdark_dynamic import (
    SuperDarkDynamic,
)
from wfi_reference_pipeline.reference_types.dark.superdark_file_batches import (
    SuperDarkBatches,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.filename_parser import FilenameParser
from wfi_reference_pipeline.utilities.logging_functions import log_info


class DarkPipeline(Pipeline):
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
    dark_pipeline = DarkPipeline("<detector string>")
    dark_pipeline.select_uncal_files()
    dark_pipeline.init_quality_control()
    dark_pipeline.prep_pipeline()
    dark_pipeline.prep_superdark()
    dark_pipeline.run_pipeline()
    dark_pipeline.pre_deliver()
    dark_pipeline.deliver()

    or

    dark_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_DARK, detector)
        self.superdark_file = None
        self.config = get_pipelines_config(REF_TYPE_DARK)

    @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("DARK SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        # files = [str(file) for file in self.ingest_path.glob("r0044401001001001001_01101_000*_WFI01_uncal.asdf")]
        files = list(
            #self.ingest_path.glob(f"r0044401001001001001_01101_0001_{self.detector}_uncal.asdf")
            #self.ingest_path.glob(f"r00444*_{self.detector}_uncal.asdf")
            self.ingest_path.glob(f"TVAC2_NOMOPS_WFIRCS_2024042501*{self.detector}_uncal.asdf")
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
            in_file = rdm.open(file)    # SAPP should this be asdf open or rdm.open

            # If save_result = True, then the input asdf file is written to disk, in the current directory, with the
            # name of the last step replacing 'uncal'.asdf
            result = DQInitStep.call(in_file, save_results=False)
            
            result = SaturationStep.call(result, save_results=False)
            result = RefPixStep.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

        logging.info("Finished PREPPING files to make DARK reference file from RFP")

    @log_info
    def prep_superdark_file(
        self,
        full_file_list=[],
        short_file_list=[],
        long_file_list=[],
        short_dark_num_reads=DARK_SHORT_NUM_READS,
        long_dark_num_reads=DARK_LONG_NUM_READS,
        sig_clip_sd_low=DARK_SIGMA_CLIP_SD_LOW,
        sig_clip_sd_high=DARK_SIGMA_CLIP_SD_HIGH,
        outfile=None
    ):
        f"""
        Prepares the superdark data file from an existing file list to be used as input for the `run_pipeline` method

            This method is designed to be flexible with pipeline runs and user interaction which is reflected in the paramter list.

            FOR AUTOMATED PIPELINE RUNS:
                No parameters are needed
                Uses self.prepped_files as file_list that gets created from `prep_pipeline` step
                Filters prepped files with detector associated with pipeline instantiation
                Assumes system default short and long dark_num_reads unless parameters clarify otherwise
                Assumes full list is adequate and does not need to filter file names

            FOR ISOLATED RUNS:
                If you are not interested in running other pipeline steps, you can utilize the parameters outlined below
                If only sending in one file list, use "full_file_list" regardless of num_reads

        Parameters
        ----------
        full_file_list: [str], optional
            A single list of all files to be processed, files must use standardized naming conventions
            This is intended for use with the pipeline architecture for special case uses
                (ie. run specific files from an existing pipeline prepped directory or re-creating a superdark)
            Mutually exclusive from short_file_list and long_file_list
        short_file_list: [str], optional
            A list of all short files to be processed, files do not require standardized naming conventions
            This is inteded for individual use where the user may not be working in the pipelines folder architecture.
                (ie. validation testing, regression testing, 3rd party users)
            These files will all be used and will not be filtered by detector
            relies on accurate short_dark_num_reads parameter
            Mutually exclusive from full_file_list
        long_file_list: [str], optional
            A list of all long files to be processed, files do not require standardized naming conventions
            These files will all be used and will not be filtered by detector
                (ie. validation testing, regression testing, 3rd party users)
            relies on accurate long_dark_num_reads parameter
            Mutually exclusive from full_file_list
        short_dark_num_reads: int, optional default={DARK_SHORT_NUM_READS}
            Number of reads for every short file
        long_dark_num_reads: int, optional default={DARK_LONG_NUM_READS}
            Number of reads for every long file

        """
        # Gather the short_dark_file_list and long_dark_file_list to send to superdark class
        if short_file_list or long_file_list:
            if full_file_list:
                raise ValueError(
                    "full_file_list parameter is mutually exclusive from short_file_list and long_file_list"
                )
            short_dark_file_list = short_file_list
            long_dark_file_list = long_file_list
        else:
            if full_file_list:
                file_list = full_file_list
            else:
                # Standard case: use the pipeline prepped files
                if len(self.prepped_files):
                    file_list = self.prepped_files
                else:
                    raise ValueError(
                        "Pipeline Files have not been prepped, run `prep_pipeline` or send in desired parameters. See Documentation for more info."
                    )  # TODO - once we have documentation add link here

            # Filter list for detector
            file_list = [file for file in file_list if self.detector in file.stem]
            short_dark_file_list, long_dark_file_list = self.extract_short_and_long_file_lists(file_list)

        if len(short_dark_file_list) == 0:
            short_dark_num_reads = 0
        if len(long_dark_file_list) == 0:
            long_dark_num_reads = 0

        # TODO - Unpack configuration
        generate_superdark_with_multiprocessing = self.config["multiprocess_superdark"]
        kwargs = {} # TODO add values to config file
        if generate_superdark_with_multiprocessing:
            logging.info("Running superdark dynamic")
            superdark = SuperDarkDynamic(
                short_dark_file_list,
                long_dark_file_list,
                self.detector,
                short_dark_num_reads=short_dark_num_reads,
                long_dark_num_reads=long_dark_num_reads,
                outfile=outfile,
            )
            kwargs = {
                "sig_clip_sd_low": sig_clip_sd_low,
                "sig_clip_sd_high": sig_clip_sd_high,
            }  # TODO - get batch sizes from config file
        else:
            logging.info("Running superdark batches")
            superdark = SuperDarkBatches(
                short_dark_file_list,
                long_dark_file_list,
                self.detector,
                short_dark_num_reads=short_dark_num_reads,
                long_dark_num_reads=long_dark_num_reads,
                outfile=outfile,
            )
            kwargs = {
                "sig_clip_sd_low": sig_clip_sd_low,
                "sig_clip_sd_high": sig_clip_sd_high,
                "short_batch_size": 4,
                "long_batch_size": 4,
            }  # TODO - get batch sizes from config file

        superdark.generate_superdark(**kwargs)
        superdark.generate_outfile()
        self.superdark_file = superdark.outfile

    @log_info
    def run_pipeline(self, file_list=None):
        logging.info("DARK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = [self.superdark_file]

        tmp = MakeDevMeta(ref_type=self.ref_type)
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_dark.mode,
            tmp.meta_dark.instrument_detector,
        )

        rfp_dark = Dark(
            meta_data=tmp.meta_dark,
            file_list=file_list,
            ref_type_data=None,
            outfile=out_file_path,
            clobber=True,
        )
        # ma_table_dict = get_ma_table_from_rtbdb(ma_table_number=3)  # TODO currently not used
        read_pattern = [[1], [2, 3], [5, 6, 7], [10]]
        rfp_dark.make_ma_table_resampled_data(read_pattern=read_pattern)
        rfp_dark.make_rate_image_from_data_cube()
        rfp_dark.generate_outfile()
        logging.info("Finished RFP to make DARK")
        print("Finished RFP to make DARK")

    def pre_deliver(self):
        pass

    def deliver(self):
        pass

    def restart_pipeline(self):
        """
        Run all steps of the pipeline.
        Redefines base class method and includes `prep_superdark_file`
        """
        self.select_uncal_files()
        self.init_quality_control()
        self.prep_pipeline()
        self.prep_superdark_file()
        self.run_pipeline()
        self.pre_deliver()
        self.deliver()

    @staticmethod
    def extract_short_and_long_file_lists(file_list):
        short_dark_file_list = []
        long_dark_file_list = []
        # TODO - Find non-test way to get short and long file list sizes, the DARK_SHORT_IDENTIFIER and DARK_LONG_IDENTIFIER wont be sustainable
        for file in file_list:
            if "TVAC" in file.name:
                short_dark_file_list.append(file)
            else:
                program_id = FilenameParser(file.name).program_id
                if DARK_LONG_IDENTIFIER == program_id:
                    long_dark_file_list.append(file)
                else:
                    short_dark_file_list.append(file)

        logging.debug("Short dark files ingested:")
        for file in short_dark_file_list:
            logging.debug(f"    {file}")
        logging.debug("Long dark files ingested:")
        for file in long_dark_file_list:
            logging.debug(f"    {file}")

        return short_dark_file_list, long_dark_file_list
