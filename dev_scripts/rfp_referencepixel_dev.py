import glob

import numpy as np

from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT
from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import (
    ReferencePixel,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import logging_functions

logging_functions.configure_logging('wfi_refpixel_example')

#####################
# Glob a set of Total Noise files
total_noise_files = glob.glob('/PATH/TO/GROUND_TESTS/TVAC1/ASDF/NOM_OPS/OTP00639_All_TV1a_R1_MCEB/Activity_1/*WFI09_uncal.asdf')[0:3]

# define the detector
detector = 'WFI09'

outfile = '/PATH/TO/scratch/rfp_referencepixel_dev_file_TVAC.asdf'

# Use dev meta maker for REFPIX
tmp = MakeDevMeta(ref_type='REFPIX')
# get the meta data
refpixel_dev_meta = tmp.meta_referencepixel

tmppath = '/PATH/TO/scratch/'
skip_first_frame = False

###################
print('-' * 80)
print('Dev to make ReferencePixel coefficienct with a list of files.')
# Instantiate rfp referencepixel object.
rfp_refpix1 = ReferencePixel(meta_data=refpixel_dev_meta,
                file_list=total_noise_files,
                outfile=outfile,
                clobber=True)
rfp_refpix1.make_referencepixel_image(tmppath=tmppath, skip_first_frame=skip_first_frame)
# Save file.
rfp_refpix1.generate_outfile()
print('Made reference file', rfp_refpix1.outfile)

#####################
print('-' * 80)
print('Dev to make ReferencePixel coefficienct with one file.')
outfile = '/PATH/TO/scratch/rfp_referencepixel_dev_file_TVAC2.asdf'

# Instantiate rfp referencepixel object.
rfp_refpix2 = ReferencePixel(meta_data=refpixel_dev_meta,
                file_list=total_noise_files[0],
                outfile=outfile,
                clobber=True)
rfp_refpix2.make_referencepixel_image(tmppath=tmppath, skip_first_frame=skip_first_frame)
rfp_refpix2.generate_outfile()


# #####################
print('-' * 80)
print('Dev to make ReferencePixel coefficients from a 3D datacube (i.e. one exposure).')
tmp_class = ReferencePixel(meta_data=refpixel_dev_meta,
                file_list=total_noise_files[0],
                outfile=outfile,
                clobber=True)
data = tmp_class.get_data_cube_from_dark_file(total_noise_files[0], skip_first_frame=skip_first_frame)

rfp_refpix3 = ReferencePixel(meta_data=refpixel_dev_meta,
                ref_type_data=data,
                outfile=outfile,
                clobber=True)
rfp_refpix3.make_referencepixel_image(tmppath=tmppath, detector_name = detector, skip_first_frame=skip_first_frame)
rfp_refpix3.generate_outfile()

#####################
print('-' * 80)
print('Dev to make ReferencePixel coefficients from a 4D datacube (i.e. multiple exposures).')
tmp_class = ReferencePixel(meta_data=refpixel_dev_meta,
                file_list=total_noise_files[0:2],
                outfile=outfile,
                clobber=True)
large_data = np.zeros((2,55, DETECTOR_PIXEL_X_COUNT, 4224))
for i, fil in enumerate(total_noise_files[0:2]):
    data = tmp_class.get_data_cube_from_dark_file(fil, skip_first_frame=skip_first_frame)
    large_data[i,:,:,:] = data

rfp_refpix4 = ReferencePixel(meta_data=refpixel_dev_meta,
                ref_type_data=large_data,
                outfile=outfile,
                clobber=True)
rfp_refpix4.make_referencepixel_image(tmppath=tmppath, detector_name = detector, skip_first_frame=skip_first_frame)
rfp_refpix4.generate_outfile()