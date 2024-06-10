# ruff: noqa

# THIS IS A TEMPORARY SCRIPT USED FOR DEVELOPMENT TESTING AND INFO SHARING BETWEEN RICK AND BRAD
# TODO - DELETE WHEN NOT NEEDED OR UPDATE INFORMATION AND INCLUDE IN TEST SUITE

import asdf, sys, os, glob, logging, time
from wfi_reference_pipeline.utilities.config_handler import get_data_files_config
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline
from wfi_reference_pipeline.constants import REF_TYPE_READNOISE
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


rfp_readnoise_pipe_all = 1
if rfp_readnoise_pipe_all == 1:
    # REFTYPE_PIPE.READNOISE

    # TODO START STANDARD INGEST

    # Step 1 - The RFP automatically quiries DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
        # I'm unclear if we are always updating the DB and then triggering off of it's contents or if we are triggering
        # off of new files being available through DAAPI and that initiates the query and then the process halts orr
        # continues depending on ref type criteria - e.g. number of files per detector
    # Step 3 - Check if criteria to make specific reference file is met

    readnoise_pipeline = ReadnoisePipeline()
    readnoise_pipeline.restart_pipeline()

rfp_readnoise_ingest_prep_only = 0
if rfp_readnoise_ingest_prep_only == 1:
    # REFTYPE_PIPE.READNOISE
    readnoise_pipeline = ReadnoisePipeline()
    readnoise_pipeline.select_uncal_files()
    readnoise_pipeline.prep_pipeline(readnoise_pipeline.uncal_files)


rfp_readnoise_pipe_only = 0
if rfp_readnoise_pipe_only == 1:

    #ingest prepped data
    prep_dir = get_data_files_config()["prep_dir"]
    prep_path = Path(prep_dir)

    # Get all readnoise files in the directory
    prepped_asdf_files = prep_path.glob(f"*READNOISE_PREPPED.asdf")
    # Convert the generator to a list if needed
    file_list = list(prepped_asdf_files)

    readnoise_pipeline = ReadnoisePipeline()
    readnoise_pipeline.run_pipeline(file_list)