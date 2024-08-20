import logging
import numpy as np
import roman_datamodels.stnode as rds
from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask

from ..reference_type import ReferenceType


class Mask(ReferenceType):
    """
    Class Mask() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_mask.asdf",
        clobber=False,
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_mask.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------
        NOTE - For parallelization only square arrays allowed.

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaMask):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaMask"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI mask reference file."

        if not (isinstance(ref_type_data, np.ndarray) and
                ref_type_data.dtype == np.uint32 and
                ref_type_data.shape == (4096, 4096)):
            raise ValueError("Mask ref_type_data must be a NumPy array of dtype uint32 and shape 4096x4096")
        else:
            self.mask = ref_type_data

        logging.debug(f"Default mask reference file object: {outfile} ")

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

    def _add_random_bad_pixels(self):
        """
        Method to add some randomly located bad pixels.
        """

        # Randomly assigning 750-850 pixels as bad pixels that dont interfere with reference pixels
        rand_num_badpixels = np.random.randint(750, 850)
        coords_x = np.random.randint(4, 4091, rand_num_badpixels)
        coords_y = np.random.randint(4, 4091, rand_num_badpixels)
        self.mask[coords_x, coords_y] += 2**0

    def calculate_error(self):
        """
        Abstract method not applicable to Mask.
        """

        pass

    def update_data_quality_array(self):
        """
        Update mask array by always ensuring the reference pixels are flagged.
        """

        self._update_mask_ref_pixels()

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the mask object from the data model.
        mask_datamodel_tree = rds.MaskRef()
        mask_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        mask_datamodel_tree['dq'] = self.mask

        return mask_datamodel_tree
