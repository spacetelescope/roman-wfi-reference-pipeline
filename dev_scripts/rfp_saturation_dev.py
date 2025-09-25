from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.saturation.saturation import Saturation
import numpy as np

print('-' * 80)

print('Dev script to make Saturation reference file with user input.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_saturation_dev_file_TVAC.asdf'
# Use dev meta maker for SATURATION.
tmp = MakeDevMeta(ref_type='SATURATION')
# Example how to change the useafter in the meta data.
tmp.meta_saturation.use_after = '2024-01-01T00:00:00.000'
# Create an empty mask array.
user_saturation_array = 47000*np.ones((4096, 4096), dtype=np.float32)
# Instantiate rfp mask object.
rfp_saturation = Saturation(meta_data=tmp.meta_saturation,
                            file_list=None,
                            ref_type_data=user_saturation_array,
                            outfile=outfile,
                            clobber=True)
# Save file.
rfp_saturation.generate_outfile()
print('Made reference file', rfp_saturation.outfile)
print('This is not how the Saturation reference files on CRDS were made. For that see below.')

print('-' * 80)

print('Dev script to make Saturation reference file for romancal development on CRDS')
outfile2 = '/grp/roman/RFP/DEV/scratch/rfp_saturation_dev_file_CRDS.asdf'
# Use dev meta maker for SATURATION
tmp2 = MakeDevMeta(ref_type='SATURATION')
# Create some sort of data. The ReferenceType base class needs to be instantiated
# with an array or a file list. The Saturation() module does not yet have the file
# list implementation.
dev_data = np.zeros((3, 3), dtype=np.float32)
# Instantiate rfp saturation object.
rfp_saturation2 = Saturation(meta_data=tmp2.meta_saturation,
                             file_list=None,
                             ref_type_data=dev_data,
                             outfile=outfile2,
                             clobber=True)
# Run make_saturation_image() to create the uniform 55000 count array.
rfp_saturation2.make_saturation_image()
# Save file.
rfp_saturation2.generate_outfile()
print('Made reference file', rfp_saturation2.outfile)
