# ruff: noqa

# THIS IS A TEMPORARY SCRIPT USED FOR DEVELOPMENT TESTING AND INFO SHARING BETWEEN RICK AND BRAD
# TODO - DELETE WHEN NOT NEEDED OR UPDATE INFORMATION AND INCLUDE IN TEST SUITE

import glob
from wfi_reference_pipeline.config.config_access import get_data_files_config
from wfi_reference_pipeline.pipelines.refpix_pipeline import RefPixPipeline

from pathlib import Path


rfp_refpix_pipe_all = 0
if rfp_refpix_pipe_all == 1:
    # REFTYPE_PIPE.REFPIX

    # Step 1 - The RFP automatically query DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
    # Step 3 - Check if criteria to make specific reference file is met

    refpix_pipeline = RefPixPipeline("WFI01")
    refpix_pipeline.restart_pipeline()


rfp_refpix_ingest_prep_only = 0
if rfp_refpix_ingest_prep_only == 1:
    # REFTYPE_PIPE.REFPIX
    refpix_pipeline = RefPixPipeline("WFI01")
    refpix_pipeline.select_uncal_files()
    refpix_pipeline.prep_pipeline(refpix_pipeline.uncal_files)


rfp_tvac_list_run_prep = 1
if rfp_tvac_list_run_prep == 1:

    files = glob.glob('/PATH/TO/GROUND_TESTS/TVAC2/ASDF/NOM_OPS/OTP00639_TotalNoiseNoEWA_TV2a_R1_MCEB/Activity_1/*WFI03*.asdf')[0:2]
    refpix_pipeline = RefPixPipeline("WFI01")
    refpix_pipeline.prep_pipeline(file_list=files)
    print(f"pipeline prepped!!")

    refpix_pipeline.run_pipeline(file_list = refpix_pipeline.prepped_files)
    print(f"run pipeline!!")
