import logging
from pathlib import Path

from wfi_reference_pipeline.constants import REF_TYPE_REFPIX
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import (
    ReferencePixel,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.logging_functions import log_info


class RefPixPipeline(Pipeline):
    """
    Derived from Pipeline Base Class.
    This is the entry point for all Reference Pixel Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save output
    run_pipeline: Process the data and create new calibration reference asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    refpix_pipeline = RefPixPipeline("<detector string>")
    refpix_pipeline.select_uncal_files()
    refpix_pipeline.prep_pipeline(refpix_pipeline.uncal_files)
    refpix_pipeline.run_pipeline(refpix_pipeline.prepped_files)
    refpix_pipeline.pre_deliver()
    refpix_pipeline.deliver()

    or

    refpix_pipeline.restart_pipeline()

    """

    def __init__(self, detector):

        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_REFPIX, detector)
        self.refpix_file = None

    @log_info
    def select_uncal_files(self):
        # Clearing from previous run
        self.uncal_files.clear()

        # TODO: note...this does not check to make sure all files are from the same detector!
        file_list = list(self.ingest_path.glob("*_uncal.asdf"))

        self.uncal_files = file_list

        logging.info(f"Ingesting {len(file_list)} files")


    @log_info
    def prep_pipeline(self, file_list=None):
        logging.info("REFPIX PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.uncal_files

        # only select files that have not been run through step 1 of the IRRC.
        # Step 2 will automatically grab all of the files, including the ones that have been previously run.
        for file in file_list:
            name = file.name + '_sums.h5'
            irrcsum_file = list((self.prep_path).glob(f'*/{name}'))

            if len(irrcsum_file) == 0:
                self.prepped_files.append(file)
            else:
                logging.info(f' >> {name} completed')
        logging.info(' >> Files used for reference pixel creation:')
        logging.info(self.prepped_files)



    @log_info
    def run_pipeline(self, file_list=None):

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.prepped_files

        tmp = MakeDevMeta(ref_type=self.ref_type)

        # TODO: how do we make sure that the WFI name is correct in the out_file_path?
        out_file_path = self.file_handler.format_pipeline_output_file_path('WIM', tmp.meta_referencepixel.instrument_detector)

        rfp_refpix = ReferencePixel(meta_data=tmp.meta_referencepixel,
                                                file_list=file_list,
                                                ref_type_data=None,
                                                outfile=out_file_path,
                                                clobber=True)

        rfp_refpix.make_referencepixel_image(tmppath=self.prep_path)

        # TODO: remove if we can figure out how to get the correct instrument detctor above for the out_file_path.
        rfp_refpix.outfile = self.file_handler.format_pipeline_output_file_path('WIM', rfp_refpix.meta_data.instrument_detector)
        rfp_refpix.generate_outfile()

        logging.info("Finished RFP to make REFPIX")
        print("Finished RFP to make REFPIX")

