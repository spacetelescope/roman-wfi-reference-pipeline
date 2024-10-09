import logging
from pathlib import Path

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from wfi_reference_pipeline.constants import REF_TYPE_DARK
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.reference_types.dark.superdark_dynamic import SuperDarkDynamic
from wfi_reference_pipeline.reference_types.dark.superdark_file_batches import SuperDarkBatches
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.logging_functions import log_info



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
    dark_pipeline.prep_pipeline()
    dark_pipeline.prep_superdark()
    dark_pipeline.run_pipeline()

    or

    dark_pipeline.restart_pipeline()

    """

    def __init__(self):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_DARK)
        self.superdark_file = None

    @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("DARK SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        # files = [str(file) for file in self.ingest_path.glob("r0044401001001001001_01101_000*_WFI01_uncal.asdf")]
        files = list(
            #self.ingest_path.glob("r0044401001001001001_01101_0001_WFI01_uncal.asdf")
            self.ingest_path.glob("r00444*_WFI01_uncal.asdf")
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
            #TODO Need to confirm steps from romancal and their functionality

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
    def prep_superdark(self, file_list=None, input_directory=None):
        """
        prepares superdark data file from an already "pipeline prepped" file list.
        By default, will use self.prepped_files (which includes the file path).
        For individual usage not part of the pipeline process, accepts file_list and input_directory
        """
        if file_list is None:
            file_list = self.prepped_files
        else:
            file_list = file_list

        # TODO - HOW TO GENERATE SHORT AND LONG FILES FROM FILE LIST?
        # UNTIL THEN USE THIS TEMPORARY GARBAGE CODE
        # Assuming short dark files contain '00444' and long dark files contain '00445'
        selected_file_list = [file for file in file_list if "WFI03" in file]
        short_dark_file_list = [file for file in selected_file_list if '00444' in file]
        print("Short dark files ingested.")
        for f in short_dark_file_list:
            print(f)
        long_dark_file_list = [file for file in selected_file_list if '00445' in file]
        print("Long dark files ingested.")
        for f in long_dark_file_list:
            print(f)


        # If true run batches, else run dynamic
        run_superdark_batches = False


        kwargs = {}
        if run_superdark_batches:
            print('Running superdark batches')
            superdark = SuperDarkBatches(input_path=input_directory,
                                        short_dark_file_list=short_dark_file_list,
                                        long_dark_file_list=long_dark_file_list)
            kwargs = {"short_batch_size": 4, "long_batch_size": 4}
        else:
            print('Running superdark dynamic')
            superdark = SuperDarkDynamic(input_path=input_directory,
                                        short_dark_file_list=short_dark_file_list,
                                        long_dark_file_list=long_dark_file_list)

        superdark.generate_superdark(**kwargs)
        superdark.generate_outfile()
        self.superdark_file = superdark.outfile

    @log_info
    def run_pipeline(self, file_list=None):
        logging.info("DARK PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.superdark_file

        tmp = MakeDevMeta(
            ref_type=self.ref_type
        )
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_dark.mode,
            tmp.meta_dark.instrument_detector,
        )

        rfp_dark = Dark(meta_data=tmp.meta_dark,
                        file_list=file_list,
                        ref_type_data=None,
                        outfile=out_file_path,
                        clobber=True
        )
        # ma_table_dict = get_ma_table_from_rtbdb(ma_table_number=3)  # TODO currently not used
        read_pattern = [[1], [2,3], [5,6,7], [10]]
        rfp_dark.make_ma_table_resampled_cube(read_pattern=read_pattern)
        rfp_dark.make_dark_rate_image()
        rfp_dark.generate_outfile()
        logging.info("Finished RFP to make DARK")
        print("Finished RFP to make DARK")


    def restart_pipeline(self):
        """
        Run all steps of the pipeline.
        Redefines base class method and includes `prep_superdark`
        """
        self.select_uncal_files()
        self.prep_pipeline()
        self.prep_superdark()
        self.run_pipeline()
