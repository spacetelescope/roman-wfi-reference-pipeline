# ruff: noqa
import asdf, sys, psutil, os, glob, logging, time
from wfi_reference_pipeline.utilities.config_handler import get_datafiles_config
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.pipeline.pipeline import Pipeline
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


write_path = '/grp/roman/RFP/DEV/scratch/'
dates = ['2020-01-01T00:00:00.000', '2021-08-01T11:11:11.111']
dates_strs = ['D1', 'D2']

# To test iterative creation of read noise reference files to suppoer
# DMS builds which have to do two useafter dates and all 18 detectors for each mode
# Make WIM read noise reference files with the RFP
make_WIM_readnoise = 0
if make_WIM_readnoise == 1:
    for d in range(0, 1):
        for det in range(1, 2):
            tmp = MakeDevMeta(ref_type='READNOISE')
            readnoise_dev_meta = tmp.meta_readnoise.export_asdf_meta()
            readnoise_dev_meta.update({'useafter': dates[d]})
            readnoise_dev_meta['instrument'].update({'detector': 'WFI' + f"{det:02d}"})
            if readnoise_dev_meta['useafter'] == '2020-01-01T00:00:00.000':
                date_str = 'D1'
            if readnoise_dev_meta['useafter'] == '2021-08-01T11:11:11.111':
                date_str = 'D2'
            sim_read_cube, _ = simulate_dark_reads(20)
            outfile = write_path + 'tmp_readnoise.asdf'
            RFPreadnoise = ReadNoise(None, meta_data=readnoise_dev_meta, outfile=outfile,
                                     clobber=True, input_data_cube=sim_read_cube)
            RFPreadnoise.comp_ramp_res_var()
            RFPreadnoise.save_readnoise()
            print('Made file -> ', outfile)
            # Set file permissions to read+write for owner, group and global
            os.chmod(outfile, 0o666)

test_flow = 0
if test_flow == 1:
    tmp = MakeDevMeta(ref_type='READNOISE')
    readnoise_dev_meta = tmp.meta_readnoise.export_asdf_meta()
    dev_rate_image = np.random.normal(loc=5, scale=1, size=(4096, 4096)).astype(np.float32)
    num_cube_reads = 20
    sim_read_cube, rate_image = simulate_dark_reads(num_cube_reads, dark_rate=1, noise_mean=15, noise_var=np.sqrt(5))

    # No data provided
    #RFPreadnoise = ReadNoise(None, meta_data=readnoise_dev_meta, outfile=outfile, clobber=True, input_data_cube=None)

    # File list and input data provide
    #RFPreadnoise = ReadNoise('filelist.txt', meta_data=readnoise_dev_meta, outfile=outfile, clobber=True, input_data_cube=sim_read_cube)

    outfile = write_path + 'roman_dev_readnoise_from_image.asdf'
    print('Testing image creation')
    RFPreadnoise = ReadNoise(None, meta_data=readnoise_dev_meta, outfile=outfile, clobber=True, input_data_cube=dev_rate_image)
    RFPreadnoise.make_readnoise_image()
    RFPreadnoise.save_readnoise()
    os.chmod(outfile, 0o666)
    print('Made file: ', RFPreadnoise.outfile)
    print('New read noise image average is ', np.mean(RFPreadnoise.readnoise_image))

    outfile = write_path + 'roman_dev_readnoise_from_cube.asdf'
    print('Testing cube creation')
    RFPreadnoise = ReadNoise(None, meta_data=readnoise_dev_meta, outfile=outfile, clobber=True, input_data_cube=sim_read_cube)
    RFPreadnoise.make_readnoise_image()
    RFPreadnoise.save_readnoise()
    os.chmod(outfile, 0o666)
    print('Made file: ', RFPreadnoise.outfile)
    print('New read noise image average is ', np.mean(RFPreadnoise.readnoise_image))

    # scratch_files = glob.glob(write_path + '*.asdf')
    # # only one detector right now
    # # need to execute by detector
    # for i in range(0, len(scratch_files)):
    #     os.remove(scratch_files[i])
    # outfile = write_path + 'roman_dev_readnoise.asdf'

#TODO for Brad development of pipeline
rfp_readnoise_pipe_all = 0
if rfp_readnoise_pipe_all == 1:
    # REFTYPE_PIPE.READNOISE

    # TODO START STANDARD INGEST

    # TODO we will need logging

    # Step 1 - The RFP automatically quiries DAAPI and downloads aka copies files from MAST
    # to somewhere on grp/roman where the RFP will know to look for new files.
    # Step 2 - Update RFP DB with new files.
        # I'm unclear if we are always updating the DB and then triggering off of it's contents or if we are triggering
        # off of new files being available through DAAPI and that initiates the query and then the process halts orr
        # continues depending on ref type criteria - e.g. number of files per detector
    # Step 3 - Check if criteria to make specific reference file is met

    print('Starting REFTYPE PREP for READNOISE')
    # Get files frominput directory
    input_dir = '/grp/roman/RFP/DEV/DAAPI_dev/asdf_files/'
    # Declare directory to write output of REFTYPE_PREP
    # files = glob.glob(input_dir + f'r0044401001001001001_01101_000*_WFI01_uncal.asdf')
    files = glob.glob(input_dir + f'r0032101001001001001_01101_0001_WFI01_uncal.asdf')
    files.sort()
    print(files)

    # TODO STOP STANDARD INGEST

    pipeline = Pipeline()
    print("begin run_readnoise")
    pipeline.run_readnoise(files)


rfp_readnoise_pipe_only = 1
if rfp_readnoise_pipe_only == 1:

    #ingest prepped data
    prep_dir = get_datafiles_config()["prep_dir"]
    prep_path = Path(prep_dir)

    # Get all readnoise files in the directory
    prepped_asdf_files = prep_path.glob(f"*READNOISE_PREPPED.asdf")
    # Convert the generator to a list if needed
    file_list = list(prepped_asdf_files)

    pipeline = Pipeline()
    pipeline.readnoise_pipe(file_list)