import logging
import numpy as np
import roman_datamodels.stnode as rds
from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask

from ..reference_type import ReferenceType

from . import mask_helpers as helper
from roman_datamodels.dqflags import pixel as dqflags

class Mask(ReferenceType):
    """
    Class Mask() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    Mask() creates the mask reference file using roman data models and
    has all necessary meta and matching criteria for delivery to CRDS.

    Example file creation commands:
    mask_obj = Mask(meta_data, ref_type_data=user_mask)
    mask_obj.make_mask_image()
    mask_obj.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_mask.asdf",
        clobber=False,
        boxwidth=15,
        sigma_stats=3.,
        dead_sigma=5.,
        max_dead_signal=0.05,
        max_low_qe_signal=0.5,
        max_open_adj_signal=1.05
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

        boxwidth: int, default = 15
            The width of the smoothing kernel.

        sigma_stats: float, default = 3.
            Sigma value for sigma clipping the master + normalized image.

        dead_sigma: float, default = 5.
            Sigma value used when determing DEAD pixels. Number of standard 
            deviations below the mean at which a pixel is considered DEAD and DO_NOT_USE.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType.
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        self.boxwidth = boxwidth
        self.sigma_stats = sigma_stats
        self.dead_sigma = dead_sigma
        self.max_dead_signal = max_dead_signal
        self.max_low_qe_signal = max_low_qe_signal
        self.max_open_adj_signal = max_open_adj_signal

        ## if they add an array will they still need filelist,,, they're compyting
        # Multiple flags are identified via normalized image
        self.normalized_image = helper.create_normalized_slope_image(
            filelist=self.file_list,
            sigma=self.sigma_stats,
            boxwidth=self.boxwidth
            )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaMask):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaMask."
            )

        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI mask reference file."

        logging.debug(f"Default mask reference file object: {outfile}.")

        # Initialize attributes.
        self.mask_image = None

        # # Module flow creating reference file.
        # if not (isinstance(ref_type_data, np.ndarray) and
        #         ref_type_data.dtype == np.uint32 and
        #         ref_type_data.shape == (4096, 4096)):
        #     raise ValueError("Mask ref_type_data must be a NumPy array of dtype uint32 and shape 4096x4096.")

        # else:
        #     logging.debug("The input 2D data array is now self.mask_image.")
        #     self.mask_image = ref_type_data
        #     logging.debug("Ready to generate reference file.")

    def make_mask_image(self):
        """
        This method is used to generate the reference file image.

        NOTE: This method is intended to be the module's internal pipeline where each method's internal
        variables and parameters are set and this is the single call to populate all attributes needed
        for the reference file data model.
        """
        # TODO add normalized image step...

        self._update_mask_ref_pixels()
        self._update_mask_dead_pixels()
        self._update_mask_low_qe_open_adj_pixels()

        self.mask_image = self.mask

    def _update_mask_ref_pixels(self):
        """
        Create array to flag the 4 pixel reference pixel border around the detector.
        """
        refpix_mask = np.zeros((4096, 4096), dtype=np.uint32)

        refpix_mask[:4, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[-4:, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, :4] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, -4:] = dqflags.REFERENCE_PIXEL.value

        self.mask += refpix_mask

    def _update_mask_dead_pixels(self):
        """
        Identify the DEAD pixels using the normalized image.
        """
        # Mean and stdev of the normalized master image
        mean_norm, stdev_norm = helper.create_image_stats(
            data=self.normalized_image,
            sigma=self.sigma_stats
        )

        # Threshold for pixel to be considered DEAD
        threshold = mean_norm - (self.dead_sigma * stdev_norm)

        # Getting the map of DEAD pixels where GOOD = 0 and DEAD = 1
        dead_mask = (self.normalized_image < threshold).astype(np.uint32)

        # Setting DEAD pixels to the actual dq flag value
        dead_mask[dead_mask == 1] = dqflags.DEAD.value

        # Padding the image with ref pixel border
        dead_mask = helper.pad_with_ref_pixels(dead_mask)

        self.mask += dead_mask

    def _update_mask_low_qe_open_adj_pixels(self):
        """
        Using the normalized image, identify LOW_QE, OPEN/ADJ pixels.
        """
        normalized_image = self.normalized_image

        # Empty arrays for the open/adj/low_qe
        low_qe_map = np.zeros(normalized_image.shape)
        open_map = np.zeros(normalized_image.shape)
        adj_map = np.zeros(normalized_image.shape)

        # A map of the locations of low signal pixels
        # TODO: use the already ID'd DEAD pix, not some arbitrary 5% number
        # If you use the 5% number, then you're gonna have to use that for DEAD ID step
        low_sig_y, low_sig_x = np.where((normalized_image > self.max_dead_signal)
                                      & (normalized_image < self.max_low_qe_signal))

        # Going through each low signal pixel and determining type
        for x, y in zip(low_sig_x, low_sig_y):

            adj_pix = normalized_image[helper.get_adjacent_pix(x, y, normalized_image)]
            all_adj = (adj_pix > self.max_open_adj_signal)

            # TODO: there currently aren't specific flags for OPEN/ADJ pixels.
            if all(all_adj):
                adj_map[y-1:y+2, x-1:x+2] = dqflags.RESERVED_5.value
                adj_map[y, x] = 0
                open_map[y, x] = dqflags.RESERVED_6.value

            else:
                low_qe_map[y, x] = dqflags.LOW_QE.value

        low_qe_map = helper.pad_with_ref_pixels(low_qe_map)
        open_map = helper.pad_with_ref_pixels(open_map)
        adj_map = helper.pad_with_ref_pixels(adj_map)

        # Adding to mask
        self.mask += low_qe_map.astype(np.uint32)
        self.mask += open_map.astype(np.uint32)
        self.mask += adj_map.astype(np.uint32)

    def _update_mask_do_not_use_pixels(self, flags=["DEAD"]):
        """
        This function adds the DO_NOT_USE flag to pixels with the following flags:
            DEAD
        This may be updated in the future with more flags.
        """
        # Gonna implement this later...
        return

    def calculate_error(self):
        """
        Abstract method not applicable to Mask.
        """

        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized by Mask().

        NOTE - Would be redundant to make_mask_image(). The attribute mask is reserved
        specifically setting the data quality arrays of other reference file types.
        """

        pass

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the mask object from the data model.
        mask_datamodel_tree = rds.MaskRef()
        mask_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        mask_datamodel_tree['dq'] = self.mask_image

        return mask_datamodel_tree
