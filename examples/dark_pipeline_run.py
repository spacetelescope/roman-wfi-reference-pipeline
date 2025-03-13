# ruff: noqa

# THIS IS A TEMPORARY SCRIPT USED FOR DEVELOPMENT TESTING AND INFO SHARING BETWEEN RICK AND BRAD
# TODO - DELETE WHEN NOT NEEDED OR UPDATE INFORMATION AND INCLUDE IN TEST SUITE

import asdf, sys, os, glob, logging, time
from wfi_reference_pipeline.utilities.config_access import get_data_files_config
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
if rfp_dark_pipe_all == 0:
    # REFTYPE_PIPE.DARK

    # Step 1 - The RFP automatically query DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
    # Step 3 - Check if criteria to make specific reference file is met

    dark_pipeline = DarkPipeline()
    dark_pipeline.restart_pipeline()


rfp_dark_ingest_prep_only = 0
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

rfp_tvac_list_run_prep = 1
if rfp_tvac_list_run_prep == 1:

    start_time = time.time()
    print("Start Time:", start_time)
    files = glob.glob("/grp/roman/GROUND_TESTS/TVAC2/ASDF/NOM_OPS/OTP00644_Darks_TV2a_R2_MCEB/Activity_1/*_WFI01*.asdf")
    dark_pipe = DarkPipeline()
    print("prepping pipeline!!")
    dark_pipe.prep_pipeline(file_list=files)
    time = time.time()
    print(f"pipeline prepped!! {time - start_time}")

    prep_dir = get_data_files_config()["prep_dir"]
    prep_path = Path(prep_dir)
    prepped_asdf_files = prep_path.glob(f"TVAC2_NOMOPS_WFIDAR_*DARK_PREPPED.asdf")
    file_list = list(prepped_asdf_files)
    dark_pipeline.prep_superdark_file(short_file_list=file_list, short_dark_num_reads=350, outfile="validate_superdark_TVAC_test_prepped_superdark_short.asdf")
    time = time.time()
    print(f"superdark prepped!! total elapsed: {time - start_time}")
# rfp_tvac_list_run_prep_superdark = 0
# if rfp_tvac_list_run_prep_superdark == 1:
#     dark_pipe = DarkPipeline()
#     fils2 = glob.glob('dark/TVAC2_NOMOPS_WFIDAR_*_PREPPED.asdf')
#     dark_pipe.prepped_files = fils2
#     dark_pipe.prep_superdark_file(short_file_list=dark_pipe.prepped_files, wfi_detector_str='01', short_dark_num_reads=350, long_dark_num_reads=0)