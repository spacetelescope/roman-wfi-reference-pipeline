import logging
import os
from multiprocessing import Pool

import numpy as np
import roman_datamodels as rdm
import roman_datamodels.stnode as rds
from astropy.convolution import Box2DKernel, convolve
from astropy.io import fits
from roman_datamodels.dqflags import pixel as dqflags

from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask

from ..reference_type import ReferenceType


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

        # Module flow creating reference file.
        if not ((isinstance(ref_type_data, np.ndarray) and
                ref_type_data.dtype == np.uint32 and
                ref_type_data.shape == (4096, 4096)) or file_list):

            raise ValueError("Mask ref_type_data must be a NumPy array of dtype uint32 and shape 4096x4096.")

        else:

            logging.debug("The input 2D data array is now self.mask_image.")
            self.mask_image = ref_type_data
            logging.debug("Ready to generate reference file.")

    def make_mask_image(self,
                        boxwidth=4,
                        dead_sigma=5.,
                        max_low_qe_signal=0.5,
                        min_open_adj_signal=1.05,
                        do_not_use_flags=["DEAD"],
                        multip=False,
                        normalized_path=None,
                        from_smoothed=True):
        """
        This method is used to generate the reference file image. We can ID bad pixels from
        dark images and flatfield images.

        See docstring for each step for more information on the flag algorithms.

        NOTE: This method is intended to be the module's internal pipeline where each method's internal
        variables and parameters are set and this is the single call to populate all attributes needed
        for the reference file data model.

        Parameters:
        -----------
        boxwidth: int, default = 4
            The width of the smoothing kernel.

        dead_sigma: float, default = 5.
            Sigma value used when determining DEAD pixels. Number of standard
            deviations below the mean at which a pixel is considered DEAD.

        max_low_qe_signal: float, default = 0.5
            Maximum normalized countrate value of a LOW_QE or OPEN pixel.

        min_open_adj_signal: float, default = 1.05
            Minimum normalized countrate value of an ADJ_OPEN pixel.
            If center pixel < max_low_qe_signal AND all adjacent pixels > min_open_adj_signal,
            then the center pixel is flagged as OPEN (instead of LOW_QE) and adjacent
            pixels are flagged as ADJ_OPEN.

        do_not_use_flags: arr of str, default = ["DEAD"]
            A list of flags whose pixels need to be marked as DO_NOT_USE.
            Used in _update_mask_do_not_use_pixels() so pixels aren't double-flagged as DNU.

        multip: bool, default = False
            When running the code in a script or ipython session, utilize
            multiprocessing's Pool map function to parallelize the code
            when calculating a super slope image.

        normalized_path: str, default = None
            If a path is specified, then the normalized image that is used
            when IDing pixels from flats is saved. If None, then no normalized
            image is saved.

        from_smoothed: bool, default = True
            If True, then the normalized image is from adaptive local
            normalization. If False, then normalized image is from mean
            normalization.

        """
        # TODO: If a user inputs ref_type_data, is that the users own mask?
        # In test_mask.py mask is created from np.zeros arr, but requires
        # that all reference pixels have value 2**31
        if self.file_list is not None:

            # TODO: we should check that the filelist is all flats/darks
            # or make sure we filter out the correct files for a given step
            self.update_mask_from_flats(filelist=self.file_list,
                                        multip=multip,
                                        from_smoothed=from_smoothed,
                                        boxwidth=boxwidth,
                                        normalized_path=normalized_path,
                                        dead_sigma=dead_sigma,
                                        max_low_qe_signal=max_low_qe_signal,
                                        min_open_adj_signal=min_open_adj_signal)

            self.update_mask_from_darks()

        # These functions can be implemented without input files
        self.update_mask_ref_pixels()
        self.set_do_not_use_pixels(do_not_use_flags=do_not_use_flags)

        # Updating the Mask object with calculated mask
        self.mask_image = self.mask

    def update_mask_from_flats(self, filelist, multip, from_smoothed, boxwidth, normalized_path, dead_sigma, max_low_qe_signal, min_open_adj_signal):
        """
        This function is used when ID'ing flags from FLAT files.
        The following flags are idenfitied:
            - DEAD: set_dead_pixels()
            - LOW_QE: set_low_qe_pixels()
            - OPEN and ADJ: set_open_adj_pixels()

        The filelist is a list of flat files; since flat images are evenly
        illuminated pixels across the entire detector, they are ideal for
        identifying low sensitivity pixels (such as DEAD).
        """
        normalized_image = self.create_normalized_image(filelist,
                                                        multip,
                                                        from_smoothed,
                                                        boxwidth)

        if normalized_path is not None:

            fits.writeto(f"{normalized_path}normalized_image.fits",
                         data=normalized_image,
                         overwrite=True)

        self.set_dead_pixels(normalized_image,
                             dead_sigma)

        self.set_low_qe_open_adj_pixels(normalized_image,
                                        max_low_qe_signal,
                                        min_open_adj_signal)

        return

    class MaskDataCube:
        def __init__(self, nz: int, degree: int = 1):
            self.nz = nz
            self.degree = degree
            self.z = np.arange(nz)

            # Precompute basis matrix and its pseudo-inverse
            self.B = np.vander(self.z, N=degree + 1, increasing=True)
            self.pinvB = np.linalg.pinv(self.B)
            self.B_x_pinvB = self.B @ self.pinvB

        def fit(self, data):
            """
            Fits the polynomial to the data.
            Returns the coefficients of the fit.
            """
            return (self.pinvB @ data.reshape(self.nz, -1)).reshape((-1, *data.shape[1:]))

        def model(self, data):
            """
            Models the data based on the polynomial fit.
            Returns the modeled data.
            """
            return (self.B_x_pinvB @ data.reshape(self.nz, -1)).reshape((-1, *data.shape[1:]))

    def _get_slope(self, file):
        """
        Extracts the slope (linear term) of the data using polynomial fitting.
        """
        with rdm.open(file) as rf:

            data = rf.data

            datacube = Mask.MaskDataCube(data.shape[0], degree=1)

            # Extract the linear coefficient
            slope = datacube.fit(data)[1]

        return slope

    def _create_super_slope_image(self, filelist, multip):
        """
        Fit a slope to each file in filelist, then average
        all slopes together to create a super slope image.
        """
        # Speed up slope calculation with Pool's map function
        if multip:

            with Pool(processes=os.cpu_count()-2) as pool:
                slopes = pool.map(self._get_slope, filelist)

        else:
            slopes = [self._get_slope(file) for file in filelist]

        super_slope_image = np.nanmean(slopes,
                                       axis=0)

        return super_slope_image

    def create_normalized_image(self, filelist, multip, from_smoothed, boxwidth):

        super_slope = self._create_super_slope_image(filelist,
                                                     multip)

        if from_smoothed:

            smoothing_kernel = Box2DKernel(boxwidth)

            smoothed_image = convolve(super_slope,
                                      smoothing_kernel,
                                      boundary="fill",
                                      fill_value=np.nanmedian(super_slope),
                                      nan_treatment="interpolate")

            return super_slope / smoothed_image

        else:

            return super_slope / np.nanmean(super_slope)

    def set_dead_pixels(self, normalized_image, dead_sigma):
        """
        Identify the DEAD pixels using the normalized image.
        A pixel is considered DEAD if it is 5 sigma below the mean
        of the normalized image.
        """
        norm_mean = np.nanmean(normalized_image)
        norm_std = np.nanstd(normalized_image)

        threshold = norm_mean - (dead_sigma * norm_std)

        # Getting the map of DEAD pixels where GOOD = 0 and DEAD = 1
        dead_mask = (normalized_image < threshold).astype(np.uint32)

        dead_mask[dead_mask == 1] = dqflags.DEAD.value

        self.mask += dead_mask

        return

    def _get_adjacent_pix(self, x_coor, y_coor, im):
        """
        Identify the pixels adjacent to a given pixel. Copied from Webb's RFP.
        This is used in set_low_qe_open_adj() function.

        Ex: note that x are the returned coordinates.
        [ ][x][ ]
        [x][o][x]
        [ ][x][ ]
        TODO: should we modify this function to return the corners too?
              Also, Tim brought up the case of two adjacent open pixels,
              currently if two open pixels are adjacent to each other then
              they're marked as LOW_QE since all four corners must be >1.05 norm im value.
        """
        y_dim, x_dim = im.shape

        if ((x_coor > 0) and (x_coor < (x_dim-1))):

            if ((y_coor > 0) and (y_coor < y_dim-1)):
                adj_x = np.array([x_coor, x_coor+1, x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor, y_coor-1, y_coor])

            elif y_coor == 0:
                adj_x = np.array([x_coor, x_coor+1, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor, y_coor])

            elif y_coor == (y_dim-1):
                adj_x = np.array([x_coor+1, x_coor, x_coor-1])
                adj_y = np.array([y_coor, y_coor-1, y_coor])

        elif x_coor == 0:

            if ((y_coor > 0) and (y_coor < y_dim-1)):
                adj_x = np.array([x_coor, x_coor+1, x_coor])
                adj_y = np.array([y_coor+1, y_coor, y_coor-1])

            elif y_coor == 0:
                adj_x = np.array([x_coor, x_coor+1])
                adj_y = np.array([y_coor+1, y_coor])

            elif y_coor == (y_dim-1):
                adj_x = np.array([x_coor+1, x_coor])
                adj_y = np.array([y_coor, y_coor-1])

        elif x_coor == (x_dim-1):

            if ((y_coor > 0) and (y_coor < y_dim-1)):

                adj_x = np.array([x_coor, x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor-1, y_coor])

            elif y_coor == 0:

                adj_x = np.array([x_coor, x_coor-1])
                adj_y = np.array([y_coor+1, y_coor])

            elif y_coor == (y_dim-1):

                adj_x = np.array([x_coor, x_coor-1])
                adj_y = np.array([y_coor-1, y_coor])

        return adj_y, adj_x

    def set_low_qe_open_adj_pixels(self, normalized_image, max_low_qe_signal, min_open_adj_signal):
        """
        Identify LOW_QE, OPEN and ADJ pixels using the normalized image.
        First, a list of coordinates of low signal pixels (defined as having a normalized
        value less than max_low_qe_signal) is created. The code then iterates through
        each of these low signal pixels, getting the four adject pixels and seeing
        if ALL of these four pixels are >1.05 norm im. If so, then this is a OPEN/ADJ
        pixel. Otherwise, then just the center is marked as LOW_QE.
        """
        low_qe_map = np.zeros((4096, 4096), dtype=np.uint32)
        open_map = np.zeros((4096, 4096), dtype=np.uint32)
        adj_map = np.zeros((4096, 4096), dtype=np.uint32)

        low_sig_y, low_sig_x = np.where(normalized_image < max_low_qe_signal)

        for x, y in zip(low_sig_x, low_sig_y):

            # Skip calculations if this is a DEAD pixel
            if self.mask[y, x] & dqflags.DEAD.value == dqflags.DEAD.value:
                continue

            adj_coor = self._get_adjacent_pix(
                x_coor=x,
                y_coor=y,
                im=normalized_image
            )

            adj_pix = normalized_image[adj_coor]
            all_adj = (adj_pix > min_open_adj_signal)

            # TODO: there currently aren't specific flags for OPEN/ADJ pixels.
            if all(all_adj):
                adj_map[y-1:y+2, x-1:x+2] = dqflags.RESERVED_5.value
                adj_map[y, x] = 0
                open_map[y, x] = dqflags.RESERVED_6.value

            else:
                low_qe_map[y, x] = dqflags.LOW_QE.value

        self.mask += low_qe_map.astype(np.uint32)
        self.mask += open_map.astype(np.uint32)
        self.mask += adj_map.astype(np.uint32)

        return

    def set_do_not_use_pixels(self, do_not_use_flags):
        """
        This function adds the DO_NOT_USE flag to pixels with flags:
            DEAD
        DO_NOT_USE pixels are excluded in subsequent pipeline processing.
        More flags may be added after further analyses.
        """
        dnupix_mask = np.zeros((4096, 4096), dtype=np.uint32)

        # Going through each DNU flag
        for flag in do_not_use_flags:

            # Bitval for the current flag
            bitval = dqflags[flag].value

            # The indices of pixels with the current iteration's flag
            flagged_pix = np.where((self.mask & bitval) == bitval)

            # Setting flagged pix to DNU bitval
            dnupix_mask[flagged_pix] = dqflags.DO_NOT_USE.value

        # Adding to mask
        self.mask += dnupix_mask.astype(np.uint32)

        return

    # TODO: Functions used when IDing flags from DARKS
    def update_mask_from_darks(self):

        return

    # Reference pixels have a static definition
    def update_mask_ref_pixels(self):
        """
        Create array to flag the 4 px reference pixel border around detector.
        """
        refpix_mask = np.zeros((4096, 4096), dtype=np.uint32)

        refpix_mask[:4, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[-4:, :] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, :4] = dqflags.REFERENCE_PIXEL.value
        refpix_mask[:, -4:] = dqflags.REFERENCE_PIXEL.value

        self.mask += refpix_mask

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
