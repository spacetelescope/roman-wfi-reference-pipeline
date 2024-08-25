from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.gain.gain import Gain
import numpy as np

print('-' * 80)

print('Dev script to make Gain reference file with user input.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_gain_dev_file_TVAC.asdf'
# Use dev meta maker for GAIN
tmp = MakeDevMeta(ref_type='GAIN')
# Example how to change the useafter in the meta data.
tmp.meta_gain.use_after = '2024-01-01T00:00:00.000'
# Create a gain array
user_gain_image = np.random.normal(loc=2, scale=0.05, size=(4096, 4096)).astype(np.float32)

# Instantiate rfp gain object.
rfp_gain = Gain(meta_data=tmp.meta_gain,
                file_list=None,
                ref_type_data=user_gain_image,
                outfile=outfile,
                clobber=True)
# Save file.
rfp_gain.generate_outfile()
print('Made reference file', rfp_gain.outfile)

print('-' * 80)
