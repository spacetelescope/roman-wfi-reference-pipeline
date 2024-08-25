from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.reference_types.dark.dark import Dark

print('-' * 80)

print('Dev to make Dark with number of reads per resultant and number of resultants for even spacing.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file_TVAC.asdf'
# Use dev meta maker for DARK
tmp = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
sim_dev_cube, sim_dev_rate_image = simulate_dark_reads(55)
# Instantiate rfp dark object.
rfp_dark = Dark(meta_data=tmp.meta_dark,
                ref_type_data=sim_dev_cube,
                outfile=outfile,
                clobber=True)
# Get rate image from data cube.
rfp_dark.make_rate_image_from_data_cube()
# Average with even spacing.
rfp_dark.make_ma_table_resampled_data(num_resultants=5, num_reads_per_resultant=1)
# Calculate error or input error array.
rfp_dark.calculate_error()
# Update the data quality array.
rfp_dark.update_data_quality_array()
# Save file.
rfp_dark.generate_outfile()
print('Made reference file', rfp_dark.outfile)

print('-' * 80)

print('Dev to make Dark from read pattern uneven spacing.')
outfile2 = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file_CRDS.asdf'
# Use dev meta to instantiate rfp dark object.
tmp2 = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
sim_dev_cube2, sim_dev_rate_image2 = simulate_dark_reads(48)
# Instantiate rfp dark object.
rfp_dark2 = Dark(meta_data=tmp2.meta_dark,
                 ref_type_data=sim_dev_cube2,
                 outfile=outfile2)
# Get rate image from data cube.
rfp_dark2.make_rate_image_from_data_cube()
# Resample and average according to read pattern.
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
rfp_dark2.make_ma_table_resampled_data(read_pattern=read_pattern)
# Update the data quality array.
rfp_dark2.update_data_quality_array()
# Save file.
rfp_dark2.generate_outfile()
print('Made reference file', rfp_dark2.outfile)


# TODO for tests
# use simulate_reads.simulate_dark_reads with a specific rate and no noise, and generate a cube of reads, then use rfp Dark to fit cube and get rate image to compare
#print(np.amax(rfp_dark.data_cube.rate_image - sim_dev_rate_image))
#print(np.amin(rfp_dark.data_cube.rate_image - sim_dev_rate_image))

print('-' * 80)

print('Dev to make Dark from read pattern with uneven spacing for RDMT-local Hack Day.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file.asdf'
# Use dev meta to instantiate rfp dark object.
tmp3 = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
sim_dev_cube3, sim_dev_rate_image3 = simulate_dark_reads(18)
# Instantiate rfp dark object.
rfp_dark3 = Dark(meta_data=tmp3.meta_dark,
                 ref_type_data=sim_dev_cube3,
                 outfile=outfile)
# Get rate image from data cube.
rfp_dark3.make_rate_image_from_data_cube()
# Resample and average according to read pattern.
read_pattern = [[1],
                [2, 3],
                [4, 5, 6, 7, 8, 9, 10]]
rfp_dark3.make_ma_table_resampled_data(read_pattern=read_pattern)
# Update the data quality array.
rfp_dark3.update_data_quality_array()
# Save file.
rfp_dark3.generate_outfile()
print('Made reference file', rfp_dark3.outfile)