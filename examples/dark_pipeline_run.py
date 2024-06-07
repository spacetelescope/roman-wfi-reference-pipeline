# ruff: noqa

# THIS IS A TEMPORARY SCRIPT USED FOR DEVELOPMENT TESTING AND INFO SHARING BETWEEN RICK AND BRAD
# TODO - DELETE WHEN NOT NEEDED OR UPDATE INFORMATION AND INCLUDE IN TEST SUITE

import asdf, sys, os, glob, logging, time
from wfi_reference_pipeline.utilities.config_handler import get_data_files_config
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.constants import REF_TYPE_DARK
from pathlib import Path
import numpy as np
import roman_datamodels as rdm

from romancal.dq_init import DQInitStep
from romancal.saturation import SaturationStep
from romancal.linearity import LinearityStep
from romancal.dark_current import DarkCurrentStep
from romancal.jump import JumpStep
from romancal.ramp_fitting import RampFitStep
from romancal.flatfield import FlatFieldStep
from romancal.assign_wcs import AssignWcsStep
from romancal.photom import PhotomStep


rfp_dark_pipe_all = 0
if rfp_dark_pipe_all == 1:
    # REFTYPE_PIPE.DARK

    # Step 1 - The RFP automatically query DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
    # Step 3 - Check if criteria to make specific reference file is met

    dark_pipeline = DarkPipeline()
    dark_pipeline.restart_pipeline()


rfp_dark_ingest_prep_only = 1
if rfp_dark_ingest_prep_only == 1:
    # REFTYPE_PIPE.DARK
    dark_pipeline = DarkPipeline()
    dark_pipeline.select_uncal_files()
    dark_pipeline.prep_pipeline(dark_pipeline.uncal_files)


rfp_dark_pipe_only = 0
if rfp_dark_pipe_only == 1:

    #ingest prepped data
    prep_dir = get_data_files_config()["prep_dir"]
    prep_path = Path(prep_dir)

    # Get all dark files in the directory
    prepped_asdf_files = prep_path.glob(f"*DARK_PREPPED.asdf")
    # Convert the generator to a list if needed
    file_list = list(prepped_asdf_files)

    #TODO why is this giving a posixpath error?
    dark_pipeline = DarkPipeline()
    dark_pipeline.run_pipeline(file_list)