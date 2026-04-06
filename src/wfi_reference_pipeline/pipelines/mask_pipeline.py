import logging
import os
import shutil
from datetime import UTC, datetime
from multiprocessing import Pool

import crds
import numpy as np
import roman_datamodels as rdm
from crds.client import api as crds_api
from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep

from wfi_reference_pipeline.constants import (
    REF_TYPE_MASK,
)
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta


class MaskPipeline(Pipeline):
    """
    Derived from the Pipeline Base Class
    This is the entry point for Mask Pipeline functionality.

    Gives user access to:
    select_uncal_files : Selecting Level 1 uncalibration asdf files
    prep_pipeline : Prepare the pipeline using romancal routines and save outputs
    run_pipeline : Process the data and create a new calibration asdf file for CRDS delivery
    restart_pipeline : Run all steps from scratch (derived from Pipeline)

    Usage:
    mask_pipeline = MaskPipeline("<detector string>")
    mask_pipeline.select_uncal_files()
    mask_pipeline.prep_pipeline()
    mask_pipeline.run_pipeline()
    mask_pipeline.pre_deliver()
    mask_pipeline.deliver()

    or

    mask_pipeline.restart_pipeline()
    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_MASK, detector)

        self.mask_file = None

    def select_uncal_files(self):
        # Clearing from previous run
        self.uncal_files.clear()

        files = list(self.ingest_path.glob("*_uncal.asdf"))
        self.uncal_files = files

        logging.info(f"Ingesting {len(files)} files: {files}")

    def _get_previous_mask_from_crds(self):
        """Get the older mask from CRDS to be used in the DQInit step."""
        logging.info(f"Downloading the latest Mask file for {self.detector} from CRDS")

        # This is where the CRDS files will be downloaded to
        os.environ["CRDS_PATH"] = self.crds_directory
        logging.info(f"CRDS_PATH: {os.environ.get('CRDS_PATH')}")

        if len(os.listdir(self.crds_directory)) > 0:
            logging.info(f"Clearing out previous Mask Pipeline's CRDS files in {self.crds_directory}")
            shutil.rmtree(self.crds_directory)
            os.makedirs(self.crds_directory)

        # TODO should this server info be in a config file?
        os.environ["CRDS_SERVER_URL"] = "https://roman-crds-test.stsci.edu"
        crds_api.set_crds_server(os.environ["CRDS_SERVER_URL"])
        logging.info(f"CRDS_SERVER_URL: {os.environ.get('CRDS_SERVER_URL')}")

        crds_context = crds.get_default_context()
        logging.info(f"CRDS context: {crds_context}")

        logging.info(f"Syncing CRDS reference files for {self.detector}...")
        params = {
            "ROMAN.META.INSTRUMENT.NAME": "WFI",
            "ROMAN.META.INSTRUMENT.DETECTOR": self.detector,
            "ROMAN.META.EXPOSURE.START_TIME": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        }

        try:
            mask_refs = crds.getreferences(params,
                                           reftypes=["mask"],
                                           context=crds_context,
                                           observatory="roman")

        except Exception as e:
            logging.info(f"Unable to retrieve the latest Mask reference file for {self.detector} using parameters {params}, {e}")

        mask_file = mask_refs["mask"]
        self.prev_mask_filepath = mask_file

        logging.info(f"Loading the previous Mask from {mask_file}")
        try:
            with rdm.open(mask_file, memmap=True) as dm:

                dq_array = np.array(dm.dq)
                self.prev_mask_image = dq_array

        except Exception as e:
            logging.info(f"Unable to load the Mask from CRDS, {e}")

    def _run_romancal(self, file):
        """
        Run romancal on a single file. Created so I can implement multiprocessing's Pool.
        """
        with rdm.open(file) as dm:

            result = DQInitStep.call(dm,
                                     save_results=False)

            result = SaturationStep.call(result,
                                         save_results=False)

            result = RefPixStep.call(result,
                                     save_results=True)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )

            result.save(path=prep_output_file_path)

        return prep_output_file_path

    def prep_pipeline(self, prep_path):
        """Prepare calibration data files by running data through select romancal steps"""
        logging.info(f"Prepping files to be run through Mask pipeline for {self.detector}")

        # This will be a temp directory for IRRC corrected files
        new_outpath = os.path.join(prep_path, "prepped_files")
        logging.info(f"Files that have been run through romancal will be in the following directory: {new_outpath}")

        if not os.path.exists(new_outpath):
            os.makedirs(new_outpath)

        self.prepped_dir = new_outpath

        # This will be a temp directory for the CRDS masks
        crds_directory = os.path.join(prep_path, "crds_mask")
        if not os.path.exists(crds_directory):
            os.makedirs(crds_directory)

        self.crds_directory = crds_directory
        logging.info(f"Previous CRDS Mask file for {self.detector} will be in {crds_directory}")

        self._get_previous_mask_from_crds()

        # Clearning up the previous run
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        try:

            with Pool() as pool:
                prepped_files = pool.map(self._run_romancal, self.uncal_files)

        except Exception as e:
            print(f"Error processing files: {e}")

        finally:
            pool.close()
            pool.join()

        self.prepped_files = prepped_files

        logging.info(f"The following files for {self.detector} have been prepped to run through the Mask pipeline: {self.prepped_files}")
        logging.info("Finished prepped the files to make Mask reference file from the RFP")

        return

    def run_pipeline(self, filelist=None):
        """Run the Mask pipeline on self.prepped_files."""
        logging.info(f"Beginning to run Mask pipeline for {self.detector}")

        if filelist is not None:
            logging.info(f"Inputted files to `filelist`. The following files will be run through the Mask: {filelist}")
        else:
            filelist = self.prepped_files
            logging.info(f"The following files will be run through the Mask: {filelist}")

        tmp = MakeDevMeta(ref_type=self.ref_type)

        out_filepath = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_mask.mode,
            tmp.meta_mask.instrument_detector,
        )

        rfp_mask = Mask(
            meta_data=tmp.meta_mask,
            file_list=filelist,
            ref_type_data=None,
            outfile=out_filepath,
            clobber=True,
        )

        logging.info("Beginning to run `make_mask_image`")
        rfp_mask.make_mask_image()

        logging.info(f"Generating the outfile Mask for {self.detector}")
        rfp_mask.generate_outfile()

        logging.info("Mask pipeline run is complete")

    def pre_deliver(self):
        pass

    def deliver(self):
        pass
