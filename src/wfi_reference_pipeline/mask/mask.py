import roman_datamodels.stnode as rds
from ..reference_type import ReferenceType
import asdf
import numpy as np


class Mask(ReferenceType):
    """
    Class Mask() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.

    Mask() inhereits reference file base class self.input_data from mask_image
    """

    def __init__(
        self,
        mask_image,
        meta_data,
        bit_mask=None,
        outfile="roman_mask.asdf",
        clobber=False
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        mask_image: numpy array; unit32
            User input mask
        meta_data: dictionary;
            Dictionary of information for reference file as required by romandatamodels.
        bit_mask: 2D integer numpy array, default=None
            A 2D data quality integer array for supplying a mask.
        outfile: string; default=roman_mask.asdf
            Filename with path for saved mask reference file.
        clobber: Boolean; default=False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        -------
        self.input_data: variable;
            The first positional variable in the Mask class instance assigned in base class ReferenceType().
            For Mask() self.input_data is a user input uint32 array.
            #TODO look at bit_mask vs mask_image possibly redundant
        """

        # Access methods of base class ReferenceType
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

    def _update_mask_ref_pixels(self):
        """
        Create array to flag the 4 pixel
        reference poxel border around the detector.
        """

        refpix_mask = np.zeros((4096, 4096), dtype=np.uint32)
        refpix_mask[:4, :] = 2**31 # apply to the top 4 rows for every column
        refpix_mask[-4:, :] = 2**31 # apply to the bottom 4 rows for every column
        refpix_mask[:, :4] = 2**31
        refpix_mask[:, -4:] = 2**31
        self.mask += refpix_mask

    def make_mask_image(self):
        """
        Add some randomly bad pixels and
        """

        self._update_mask_ref_pixels()
        # randomly assigning 750-850 pixels as bad pixels that dont interfere with reference pixels
        rand_num_badpixels = np.random.randint(750, 850)
        coords_x = np.random.randint(4, 4091, rand_num_badpixels)
        coords_y = np.random.randint(4, 4091, rand_num_badpixels)
        self.mask[coords_x, coords_y] += 2**0

        # Use supplied mask image if it was provided.
        if self.input_data is not None:
            if isinstance(self.input_data, np.ndarray) and self.input_data.dtype == np.uint32 \
                    and self.input_data.shape == (4096, 4096):
                self.mask += self.input_data
            else:
                raise ValueError("Input mask is not type unit32 or size 4096x4096.")

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
