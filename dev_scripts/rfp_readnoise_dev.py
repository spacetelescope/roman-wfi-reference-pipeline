import os
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
import numpy as np

print('Dev to make ReadNoise with an input image.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_readnoise_dev_file.asdf'
# Use dev meta to instantiate rfp readnoise object.
tmp = MakeDevMeta(ref_type='READNOISE')
# Create user input rate image
user_rate_image = np.random.normal(loc=5, scale=1, size=(4096, 4096)).astype(np.float32)
# Instantiate rfp readnoise object.
rfp_readnoise = ReadNoise(meta_data=tmp.meta_readnoise,
                          ref_type_data=user_rate_image,
                          outfile=outfile,
                          clobber=True)
# Save file.
rfp_readnoise.generate_outfile()
# Set file permissions to read+write for owner, group and global.
os.chmod(outfile, 0o666)


print('Dev to make ReadNoise with a simulated dark cube.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_readnoise_dev_file.asdf'
# Use dev meta to instantiate rfp readnoise object.
tmp = MakeDevMeta(ref_type='READNOISE')
# Set simulated cube readnoise variance for testing of ReadNoise.
sim_readnoise_var = 5
# Create simulated data cube.
sim_dev_cube, dev_rate_image = simulate_dark_reads(40, dark_rate=1, noise_mean=15, noise_var=np.sqrt(sim_readnoise_var))
# Instantiate rfp readnoise object.
rfp_readnoise = ReadNoise(meta_data=tmp.meta_readnoise,
                          ref_type_data=sim_dev_cube,
                          outfile=outfile,
                          clobber=True)
# Get rate image from data cube.
rfp_readnoise.make_rate_image_from_data_cube()
# Make readnoise image from data cube.
rfp_readnoise.make_readnoise_image()
# Save file.
rfp_readnoise.generate_outfile()
# Set file permissions to read+write for owner, group and global.
os.chmod(outfile, 0o666)

#TODO for readnoise testing
print(np.mean(rfp_readnoise.readnoise_image),np.median(rfp_readnoise.readnoise_image), sim_readnoise_var)
