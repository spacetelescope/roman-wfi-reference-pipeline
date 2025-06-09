# ruff: noqa

# THIS IS A TEMPORARY SCRIPT USED FOR DEVELOPMENT TESTING AND INFO SHARING BETWEEN RICK AND BRAD
# TODO - DELETE WHEN NOT NEEDED OR UPDATE INFORMATION AND INCLUDE IN TEST SUITE

import glob
from wfi_reference_pipeline.config.config_access import get_data_files_config
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from pathlib import Path


rfp_dark_pipe_all = 0
if rfp_dark_pipe_all == 1:
    # REFTYPE_PIPE.DARK

    # Step 1 - The RFP automatically query DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
    # Step 3 - Check if criteria to make specific reference file is met

    dark_pipeline = DarkPipeline("WFI01")
    dark_pipeline.restart_pipeline()


rfp_dark_ingest_prep_only = 0
if rfp_dark_ingest_prep_only == 1:
    # REFTYPE_PIPE.DARK
    dark_pipeline = DarkPipeline("WFI01")
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

    dark_pipeline = DarkPipeline("WFI01")
    dark_pipeline.run_pipeline(file_list)

rfp_tvac_list_run_prep = 1
if rfp_tvac_list_run_prep == 1:

    files = glob.glob("/grp/roman/GROUND_TESTS/TVAC2/ASDF/NOM_OPS/OTP00644_Darks_TV2a_R2_MCEB/Activity_1/*_WFI01*.asdf")
    dark_pipe = DarkPipeline("WFI01")
    dark_pipe.prep_pipeline(file_list=files)
    print(f"pipeline prepped!!")

    prep_dir = get_data_files_config()["prep_dir"]
    prep_path = Path(prep_dir)
    prepped_asdf_files = prep_path.glob(f"TVAC2_NOMOPS_WFIDAR_*DARK_PREPPED.asdf")
    file_list = list(prepped_asdf_files)
    dark_pipe.prep_superdark_file(short_file_list=file_list, short_dark_num_reads=350, outfile="validate_superdark_TVAC_test_prepped_superdark_short.asdf")
    print(f"superdark prepped!!")
