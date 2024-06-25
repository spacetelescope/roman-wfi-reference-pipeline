from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.mask.mask import Mask
import numpy as np

print('Dev to make Mask reference file.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_mask_dev_file.asdf'
# Use dev meta to instantiate rfp mask object.
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
# Update data quality array to have mask flag reference pixels.
rfp_mask.update_data_quality_array()
# Save file.
rfp_mask.generate_outfile()
