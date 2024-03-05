import logging
import roman_datamodels as rdm

from pathlib import Path
from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from romancal.linearity import LinearityStep

from wfi_reference_pipeline.constants import REF_TYPE_READNOISE
from wfi_reference_pipeline.pipeline import Pipeline
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.utilities.file_handler import format_prep_output_file_path, format_calibrated_output_file_path
from wfi_reference_pipeline.utilities.logging_functions import log_info



class ReadnoisePipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Readnoise Pipeline functionality
    Gives user access to:
        select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
        prep_pipeline : Preparing the pipeline using romancal routines and save output
        run_pipeline: Calibrate the data and create new asdf file for CRDS delivery
        restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
        readnoise_pipeline = ReadnoisePipeline()
        readnoise_pipeline.select_uncal_files()
        readnoise_pipeline.prep_pipeline(readnoise_pipeline.uncal_files)
        readnoise_pipeline.run_pipeline(readnoise_pipeline.prepped_files)

        or

        readnoise_pipeline.restart_pipeline()

    """

    def __init__(self):
        # Initialize baseclass from here for access to this class name
        super().__init__()

        # TODO input ref type specific configuration

    @log_info
    def select_uncal_files(self):

        self.uncal_files.clear()
        logging.info("READNOISE SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        files = [str(file) for file in self.ingest_path.glob("r0044401001001001001_01101_000*_WFI01_uncal.asdf")]
        # files = list(self.ingest_path.glob("r0032101001001001001_01101_0001_WFI01_uncal.asdf"))

        self.uncal_files = files
        logging.info(f"Ingesting {len(files)} Files: {files}")

    @log_info
    def prep_pipeline(self, file_list=None):

        # TODO - Remove existing READNOISE files from prepped directory before running
        logging.info("READNOISE PREP")
        # Convert file_list to a list of Path type files
        file_list = list(map(Path, file_list))
        self.prepped_files.clear()
        # self._datamodels_prepped.clear()
        for file in file_list:
            logging.info("OPENING - " + file.name)
            in_file = rdm.open(file)

            # If save_result = True, then the input asdf file is written to disk, in the current directory, with the
            # name of the last step replacing 'uncal'.asdf
            # TODO - make the save_results out directory configurable
            result = DQInitStep.call(in_file, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = LinearityStep.call(result, save_results=False)

            prep_output_file_path = format_prep_output_file_path(self.prep_path, result.meta.filename[:-10], "READNOISE")  # TODO standardize the extraction of the filename
            result.save(path=prep_output_file_path)

            # self._datamodels_prepped.append(result)
            self.prepped_files.append(prep_output_file_path)

        logging.info('Finished PREPPING files to make READNOISE reference file from RFP')

    @log_info
    def run_pipeline(self, file_list):

        # TODO load config file with defaults or other params to run pipeline
        # TODO I dont know if ReadNoise will need to be parallelized but dark might be, my thinking is that the class
        # would be instantiated for the whole detector and then outside of here, sub arrays are organized and send individually
        # into methods and then the array is reassmebled to save to disk
        # TODO Quality Check before writing file to disk. Where should a QC check be done?

        logging.info('READNOISE PIPE')

        file_list = list(map(Path, file_list))
        tmp = MakeDevMeta(ref_type=REF_TYPE_READNOISE)  # TODO replace with MakeMeta which gets actual information from files
        readnoise_dev_meta = tmp.meta_readnoise.export_asdf_meta()
        out_file_path = format_calibrated_output_file_path(self.calibrated_out_path, 'roman_dev_readnoise_from_DAAPI_dev_files.asdf') # TODO make this name generic

        rfp_readnoise = ReadNoise(file_list, meta_data=readnoise_dev_meta, outfile=out_file_path, clobber=True)
        rfp_readnoise.make_readnoise_image()

        rfp_readnoise.save_readnoise()
        out_file_path.chmod(0o666)
        logging.info('Finished RFP to make READNOISE')
        print('Finished RFP to make READNOISE')


