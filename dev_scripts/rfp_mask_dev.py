from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.mask.mask import Mask
import numpy as np

print('-' * 80)

print('Dev script to make Mask reference file with user input.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_mask_dev_file_TVAC.asdf'
# Use dev meta maker for MASK
tmp = MakeDevMeta(ref_type='MASK')
# Example how to change the useafter in the meta data.
tmp.meta_mask.use_after = '2024-01-01T00:00:00.000'
# Create an empty mask array.
user_mask = np.zeros((4096, 4096), dtype=np.uint32)
# Set a single pixel to a dq flag of 2^2 = 4
user_mask[5, 5] = 4
# Instantiate rfp mask object.
rfp_mask = Mask(meta_data=tmp.meta_mask,
                file_list=None,
                ref_type_data=user_mask,
                outfile=outfile,
                clobber=True)
# Save file.
rfp_mask.generate_outfile()
print('Made reference file', rfp_mask.outfile)
print('Methods inside Mask() were not run so reference pixels are not marked in the mask.')

print('-' * 80)

print('Dev script to make Mask reference file for romancal development on CRDS')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_mask_dev_file_CRDS.asdf'
# Use dev meta maker for MASK
tmp2 = MakeDevMeta(ref_type='MASK')
# Create an empty mask array.
user_mask2 = np.zeros((4096, 4096), dtype=np.uint32)
# Instantiate rfp mask object.
# Need some type of ref_type_data as input to pass data check tests.
rfp_mask2 = Mask(meta_data=tmp.meta_mask,
                 file_list=None,
                 ref_type_data=user_mask2,
                 outfile=outfile,
                 clobber=True)
# Need to add reference pixels and some random bad pixels.
# These methods are called inside the make_mask_image() method.
rfp_mask2.make_mask_image()
# Now the self.mask_image has the user input plus random bad pixels and the reference pixel flags set.
# Save file.
rfp_mask2.generate_outfile()
print('Made reference file', rfp_mask2.outfile)
print('Method make_mask_image() creates a mask with reference pixels and randomly located bad pixels.')
