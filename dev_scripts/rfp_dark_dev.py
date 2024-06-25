import os
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import simulate_reads
from wfi_reference_pipeline.reference_types.dark.dark import Dark
import numpy as np

print('Dev to make Dark with number of reads per resultant and number of resultants for even spacing.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file.asdf'
# Use dev meta to instantiate rfp dark object.
tmp = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
sim_dev_cube, sim_dev_rate_image = simulate_reads.simulate_dark_reads(48)
# Instantiate rfp dark object.
rfp_dark = Dark(meta_data=tmp.meta_dark,
                ref_type_data=sim_dev_cube,
                outfile=outfile,
                clobber=True)
# Get rate image from data cube.
rfp_dark.make_rate_image_from_data_cube()
# Average with even spacing.
rfp_dark.make_ma_table_resampled_data(num_resultants=8, num_reads_per_resultant=6)
# Save file.
rfp_dark.generate_outfile()


print('Dev to make Dark with from read pattern uneven spacing.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file.asdf'
# Use dev meta to instantiate rfp dark object.
tmp = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
sim_dev_cube2, sim_dev_rate_image2 = simulate_reads.simulate_dark_reads(48)
# Instantiate rfp dark object.
rfp_dark = Dark(meta_data=tmp.meta_dark,
                ref_type_data=sim_dev_cube2,
                outfile=outfile)
# Get rate image from data cube.
rfp_dark.make_rate_image_from_data_cube()
# Average with even spacing.
read_pattern = [
    [1],
    [2, 3],
    [4, 5, 6, 7],
    [9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26],
    [28, 29, 30, 31, 32, 33],
    [37, 38, 39, 40, 41, 42],
    [44, 45, 46, 47],
    [48]
]
rfp_dark.make_ma_table_resampled_data(read_pattern=read_pattern)
# Save file.
rfp_dark.generate_outfile()

#TODO for tests
# use simulate_reads.simulate_dark_reads with a specific rate and no noise, and generate a cube of reads, then use rfp Dark to fit cube and get rate image to compare
print(np.amax(rfp_dark.data_cube.rate_image - sim_dev_rate_image))
print(np.amin(rfp_dark.data_cube.rate_image - sim_dev_rate_image))
