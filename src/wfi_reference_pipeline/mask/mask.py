import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np


class Mask(ReferenceFile):
    """
    Class Mask() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.
    """

    def __init__(
        self,
        mask_image,
        meta_data,
        bit_mask=None,
        outfile="roman_mask.asdf",
        clobber=False
    ):

        # Access methods of base class ReferenceFile
        super().__init__(
            mask_image,
            meta_data,
            bit_mask=bit_mask,
            clobber=clobber,
            make_mask=True,
        )

        # Update metadata with file type info if not included.
        if "description" not in self.meta.keys():
            self.meta["description"] = "Roman WFI mask reference file."
        if "reftype" not in self.meta.keys():
            self.meta["reftype"] = "MASK"

        # Initialize attributes
        self.outfile = outfile
        self.mask = mask_image

    def make_mask_image(self):
        """


        """

        # set reference pixel border
        mask = np.zeros((4096, 4096), dtype=np.uint32)
        mask[:4, :] += 2**31 # apply to the top 4 rows for every column
        mask[-4:, :] += 2**31 # apply to the bottom 4 rows for every column
        mask[:, :4] += 2**31
        mask[:, -4:] += 2**31

        # randomly assigning 750-850 pixels as bad pixels
        rand_num_badpixels = np.random.randint(750, 850)
        coords_x = np.random.randint(4, 4091, rand_num_badpixels) # make sure this does not overlap the border
        coords_y = np.random.randint(4, 4091, rand_num_badpixels)
        mask[coords_x, coords_y] += 2**0

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        mask_datamodel_tree = rds.MaskRef()
        mask_datamodel_tree['meta'] = self.meta
        mask_datamodel_tree['dq'] = self.mask

        return mask_datamodel_tree

    def save_mask(self, datamodel_tree=None):
        """
        The method save_mask writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
