import os
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import simulate_reads
from wfi_reference_pipeline.reference_types.dark.dark import Dark

outfile = '/grp/roman/RFP/DEV/scratch/rfp_dark_dev_file.asdf'
# Use dev meta to instantiate rfp dark object.
tmp = MakeDevMeta(ref_type='DARK')
# Simulate a cube of dark reads.
simulated_read_cube, _ = simulate_reads.simulate_dark_reads(48)
# Instantiate rfp dark object.
rfp_dark = Dark(meta_data=tmp.meta_dark,
                ref_type_data=simulated_read_cube,
                outfile=outfile)
# Get rate image from data cube.
rfp_dark.get_dark_rate_image_from_data_cube()
# Average with even spacing.
rfp_dark.make_ma_table_resampled_data(num_resultants=8, num_reads_per_resultant=6)
# Save file.
rfp_dark.generate_outfile()
# Set file permissions to read+write for owner, group and global.
os.chmod(outfile, 0o666)
