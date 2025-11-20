import glob
import logging
import os
from multiprocessing import Pool

import roman_datamodels as rdm
from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep

from wfi_reference_pipeline.constants import (
    REF_TYPE_MASK,
)
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# TODO: FilenameParser will be useful when making files for different detectors since can split by SCA
# from wfi_reference_pipeline.utilities.filename_parser import FilenameParser


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

    def select_uncal_files(self, filelist):
        # Clearing from previous run
        self.uncal_files.clear()

        # TODO: how would users go about specifying which detector they want
        # to focus on? The paths here are specified in the config file so idk
        files = list(self.ingest_path.glob("*_uncal.asdf"))

        self.uncal_files = files

        logging.info(f"Ingesting {len(files)} files: {files}")

    def run_romancal(self, file, outpath):
        """
        Run romancal on a single file. Created so I can implement multiprocessing's Pool.
        """
        with rdm.open(file) as f:
            dq_data = DQInitStep.call(f)

            _ = RefPixStep.call(dq_data, save_results=True, output_dir=outpath)

        return

    def prep_pipeline(self, prep_path):
        """Prepare calibration data files by running data through select romancal steps"""
        # This will be a temp directory for IRRC corrected files
        new_outpath = f"{prep_path}irrc_corr/"

        if not os.path.exists(new_outpath):
            os.path.makedirs(new_outpath)

        args = [(file, new_outpath) for file in self.uncal_files]

        with Pool() as pool:
            _ = pool.starmap(self.run_romancal, args)

        self.prepped_files = glob.glob(f"{new_outpath}**asdf")

        return

    def run_pipeline(self, outfile_path, file_list):
        tmp = MakeDevMeta(ref_type=self.ref_type)

        rfp_mask = Mask(
            meta_data=tmp.meta_mask,
            file_list=self.prepped_files,
            ref_type_data=None,
            outfile=outfile_path,
            clobber=True,
        )

        rfp_mask.make_mask_image()
        rfp_mask.generate_outfile()
