import os
import logging
import roman_datamodels as rdm

from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from romancal.linearity import LinearityStep

from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.utilities.file_handler import get_calibrated_output_file_path, get_prep_output_file_path
from wfi_reference_pipeline.utilities import logging_functions

logging_functions.configure_logging("Pipeline")

class Pipeline():

    def __init__(self):
        self.readnoise_files_prepped = []
        self.readnoise_datamodels_prepped = []

    def readnoise_prep(self, file_list):
        logging.info("readnoise prep")
        self.readnoise_files_prepped.clear()
        self.readnoise_datamodels_prepped.clear()
        for file in file_list:
            logging.info("OPENING - " + file)
            in_file = rdm.open(file)

            # If save_result = True, then the input asdf file is written to disk, in the current directory, with the
            # name of the last step replacing 'uncal'.asdf
            result = DQInitStep.call(in_file, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = LinearityStep.call(result, save_results=True)

            output_prep_file = get_prep_output_file_path(result.meta.filename[:-10], "READNOISE")  # TODO standardize the extraction of the filename
            result.save(path=output_prep_file)

            self.readnoise_datamodels_prepped.append(result)
            self.readnoise_files_prepped.append(output_prep_file)

        logging.info('Finished PREPPING input DARK files to make READNOISE reference file from RFP')

    def readnoise_pipe(self, file_list):

        # TODO load config file with defaults or other params to run pipeline
        # TODO I dont know if ReadNoise will need to be parallelized but dark might be, my thinking is that the class
        # would be instantiated for the whole detector and then outside of here, sub arrays are organized and send individually
        # into methods and then the array is reassmebled to save to disk
        # TODO Quality Check before writing file to disk. Where should a QC check be done?

        logging.info('Starting RFP to make READNOISE')
        tmp = MakeDevMeta(ref_type='READNOISE')  # TODO replace with MakeMeta which gets actual information from files
        readnoise_dev_meta = tmp.meta_readnoise.export_asdf_meta()
        outfile = get_calibrated_output_file_path('roman_dev_readnoise_from_DAAPI_dev_files.asdf')

        rfp_readnoise = ReadNoise(file_list, meta_data=readnoise_dev_meta, outfile=outfile, clobber=True)
        rfp_readnoise.make_readnoise_image()

        rfp_readnoise.save_readnoise()
        os.chmod(outfile, 0o666)

    def run_readnoise(self, file_list):
        self.readnoise_prep(file_list)
        self.readnoise_pipe(self.readnoise_files_prepped) # TODO change this to self.readnoise_datamodels_prepped when code can handle it.

