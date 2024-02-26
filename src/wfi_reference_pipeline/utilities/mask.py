import roman_datamodels.stnode as rds
from ..reference_file import ReferenceFile
import asdf
import numpy as np


class Mask(ReferenceFile):
    """
    Class Mask() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.
    """


    def __init__(self, mask_image, meta_data, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_maskfile.asdf'

        # Access methods of base class ReferenceFile
        super(Mask, self).__init__(mask_image, meta_data, clobber=clobber)

        # Update metadata with mask file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI mask reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'MASK'
        else:
            pass

    def make_maskfile(self):
        """
        The method make_maskfile() generates a mask asdf file.

        Parameters
        ----------

        Outputs
        -------
        af: asdf file tree: {meta, dq}
            meta:
            dq: mask - data quality array
                masked reference pixels flagged 2**31
                masked bad pixels flagged 2**1
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the read noise object from the data model.
        maskfile = rds.MaskRef()
        maskfile['meta'] = self.meta

        # set reference pixel border
        mask = np.zeros((4096, 4096), dtype=np.uint32)
        mask[:4, :] += 2**31 # apply to the top 4 rows for every column
        mask[-4:, :] += 2**31 # apply to the bottom 4 rows for every column
        mask[:, :4] += 2**31
        mask[:, -4:]+= 2**31

        # randomly assigning 750-850 pixels as bad pixels
        rand_num_badpixels = np.random.randint(750, 850)
        coords_x = np.random.randint(4, 4091, rand_num_badpixels) # make sure this does not overlap the border
        coords_y = np.random.randint(4, 4091, rand_num_badpixels)
        mask[coords_x, coords_y] += 2**0

        # udpate dq array to have mask
        maskfile['dq'] = mask
        # Mask files do not have data or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': maskfile}
        af.write_to(self.outfile)
