from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
import numpy as np

print('-' * 80)

print('Dev script to make Read Noise reference file with user input.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_readnoise_dev_file_TVAC.asdf'
# Use dev meta maker for READNOISE
tmp = MakeDevMeta(ref_type='READNOISE')
# Example how to change the useafter in the meta data.
tmp.meta_readnoise.use_after = '2024-01-01T00:00:00.000'
# Create user input rate image - imagine this is your TVAC analysis read noise image.
user_rate_image = np.random.normal(loc=5, scale=1, size=(4096, 4096)).astype(np.float32)
# Instantiate rfp readnoise object.
rfp_readnoise = ReadNoise(meta_data=tmp.meta_readnoise,
                          ref_type_data=user_rate_image,
                          outfile=outfile,
                          clobber=True)
# When supplying a 2D image to ReadNoise, you can create a CRDS ready file without running
# the make_readnoise_image() method.
# Save file.
rfp_readnoise.generate_outfile()
print('Made reference file', rfp_readnoise.outfile)

print('-' * 80)

print('Dev script to make Read Noise reference file for romancal development on CRDS')
print('Give ReadNoise a simulated dark cube or an exposure up the ramp.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_readnoise_dev_CRDS.asdf'
# Use dev meta maker for READNOISE
tmp2 = MakeDevMeta(ref_type='READNOISE')
# Set simulated cube read noise variance for testing ReadNoise().
sim_readnoise_std = 5
# Create simulated data cube. Adjust parameters as desired.

sim_dev_cube, dev_rate_image = simulate_dark_reads(55,
                                                   dark_rate=.001,
                                                   dark_rate_var=0.0,
                                                   num_hot_pix=0,
                                                   num_warm_pix=0,
                                                   num_dead_pix=0,
                                                   noise_mean=200,
                                                   noise_std=sim_readnoise_std)
# Instantiate rfp readnoise object.
rfp_readnoise2 = ReadNoise(meta_data=tmp2.meta_readnoise,
                           ref_type_data=sim_dev_cube,
                           outfile=outfile,
                           clobber=True)
# Make readnoise image from data cube.
# The make_readnoise_image() is a mini internal module pipeline that runs the steps necessary
# to create the readnoise image from a data cube.
rfp_readnoise2.make_readnoise_image()
# Save file.
rfp_readnoise2.generate_outfile()
print('Made reference file', rfp_readnoise2.outfile)

#TODO for readnoise testing
# Determine minimum number of reads needed to recover simulated variance when testing the compute
# variance of the residuals method to estimate read noise.
print(np.mean(rfp_readnoise.readnoise_image), np.median(rfp_readnoise.readnoise_image), sim_readnoise_std)
