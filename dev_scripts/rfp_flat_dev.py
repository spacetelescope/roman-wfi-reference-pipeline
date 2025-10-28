import numpy as np

from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import simulate_reads

print('-' * 80)

print('Dev script to make Flat reference file with user input.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_flat_dev_file_TVAC.asdf'
# Use dev meta maker for FLAT
tmp = MakeDevMeta(ref_type='FLAT')
# Example how to change the useafter in the meta data.
tmp.meta_flat.use_after = '2024-01-01T00:00:00.000'
# Simulate a cube of flat reads.
_, sim_dev_rate_image = simulate_reads.simulate_flat_reads(10)
# Flatten the rate image to be input into Flat().
dev_flat_rate_image = sim_dev_rate_image / np.mean(sim_dev_rate_image)
# Instantiate rfp flat object.
rfp_flat = Flat(meta_data=tmp.meta_flat,
                ref_type_data=dev_flat_rate_image,
                outfile=outfile,
                clobber=True)
# The flat error array and dq mask are all zero. The current
# rfp flat object can be written to disk by running rfp_flat.generate_outfile() now.
rfp_flat.generate_outfile()
print('Made reference file', rfp_flat.outfile)

print('-' * 80)

print('Dev script to make Flat reference file from input data cube and made to be like others'
      ' on CRDS with non-zero error array and randomly located loq qe pixels in the dq mask.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_flat_dev_file_CRDS.asdf'
# Use dev meta maker for FLAT
tmp2 = MakeDevMeta(ref_type='FLAT')
# Example how to change the useafter in the meta data.
tmp2.meta_flat.use_after = '2024-01-01T00:00:00.000'
# Simulate a cube of flat reads.
sim_dev_cube2, _ = simulate_reads.simulate_flat_reads(10)
# Instantiate rfp flat object.
rfp_flat2 = Flat(meta_data=tmp.meta_flat,
                 ref_type_data=sim_dev_cube2,
                 outfile=outfile,
                 clobber=True)
# Must run make_flat_image()
rfp_flat2.make_flat_image()
# If no error array is provided, some random error is added to flat_error.
rfp_flat2.calculate_error()
# Update the data quality array and add low_qe_pixels.
rfp_flat2.update_data_quality_array(add_low_qe_pixels=True)
# Save file.
rfp_flat2.generate_outfile()
print('Made reference file', rfp_flat2.outfile)

print('-' * 80)
