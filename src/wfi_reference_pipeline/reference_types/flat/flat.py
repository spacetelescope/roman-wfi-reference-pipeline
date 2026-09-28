import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import asdf
import numpy as np
from astropy import units as u
from roman_datamodels.datamodels import FlatRefModel

from wfi_reference_pipeline.constants import (
    SCI_PIXEL_X_COUNT,
    SCI_PIXEL_Y_COUNT,
    WFI_TYPE_IMAGE,
)
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat

from ..reference_type import ReferenceType


class Flat(ReferenceType):
    """
    Class Flat() inherits the ReferenceType() base class methods where
    static meta data for all reference file types are written.

    The class Flat() ingests a list of files and finds all exposures with the same filter
    within some maximum date range. Fit ramps to all available filter
    cubes are used to generate flat rate images and average together and normalize
    to produce the filter dependent flat rate image.

    Example file creation commands:
    With user array.
    flat_obj = Flat(meta_data, ref_type_data=flattened_array)
    flat_obj.generate_outfile()

    With user cube input.
    flat = Flat(meta_data, ref_type_data=data_cube)
    flat.make_flat_image()
    flat.calculate_error()
    flat.update_data_quality_array()
    flat.generate_outfile()
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_flat.asdf",
            clobber=False,
            file_map=None
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
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        file_map: dict
            A dictionary structured as: {(target_sca_int, primary_file_path): exact_matching_file_path}
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Default bit mask size of 4088x4088 for flat is size of science array
        # and must be provided if not bit_mask to instantiate properly in base class.
        if bit_mask is None:
            bit_mask = np.zeros((SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT), dtype=np.uint32)

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
        if not isinstance(meta_data, WFIMetaFlat):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaFlat"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI flat reference file."

        logging.debug(f"Default flat reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.flat_image = None  # The attribute 'data' in data model.
        self.flat_error = None  # The attribute assigned to the flat['err'].
        self.num_files = 0
        self.file_map = file_map
        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.num_files = len(self.file_list)
            # Must make_flat_image() to finish creating reference file.
        else:
            if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            # Only access data from quantity object.
            if isinstance(ref_type_data, u.Quantity):
                ref_type_data = ref_type_data.value
                logging.debug(
                    "Quantity object detected. Extracted data values.")

            dim = ref_type_data.shape
            if len(dim) == 2:
                logging.debug(
                    "The input 2D data array is now self.flat_image.")
                self.flat_image = ref_type_data.astype(np.float32)
                logging.debug(
                    "Initializing flat error array with all zeros."
                )
                self.flat_error = np.zeros((SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT), dtype=np.float32)
                logging.debug("Ready to generate reference file.")
            elif len(dim) == 3:
                logging.debug(
                    "User supplied 3D data cube to make flat reference file."
                )
                self.data_cube = self.FlatDataCube(
                    ref_type_data, WFI_TYPE_IMAGE
                )
                # Must call make_flat_image() to finish creating reference file.
                logging.debug(
                    "Must call make_flat_image() to finish creating reference file."
                )
            else:
                raise ValueError(
                    "Input data is not a valid numpy array of dimension 2 or 3."
                )

    def make_flat_image(self):
        """
        This method is used to generate the reference file image from the file list or a data cube.

        NOTE: This method is intended to be the module's internal pipeline where each method's internal
        variables and parameters are set and this is the single call to populate all attributes needed
        for the reference file data model.

        The flat reference file data model has:
            data = self.flat_image
            err = self.flat_error
            dq = self.dq_mask
        Additional method calls must be run to populate initialized arrays:
            self.calculate_error()
            self.update_data_quality_array()
        """

        if self.file_list:
            logging.debug(
                "Making flat_image from average of rate images from file list."
            )
            # Run and populate flat image array and error
            self.make_flat_from_files(calc_error=True)
        else:
            logging.debug(
                "Making flat_image from data cube."
            )
            self.make_rate_image_from_data_cube()
            self.flat_image = self.data_cube.rate_image / \
                np.mean(self.data_cube.rate_image)

        logging.debug(
            "Initializing flat error array with all zeros. Run calculate_error()."
        )
        self.flat_error = np.zeros((SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT), dtype=np.float32)
        logging.debug("Ready to generate reference file.")

    def make_rate_image_from_data_cube(self, fit_order=1):
        """
        Method to fit the data cube. Intentional method call to specific fitting order to data.

        Parameters
        ----------
        fit_order: integer; Default=None
            The polynomial degree sent to data_cube.fit_cube.

        Returns
        -------
        self.data_cube.rate_image: object;
        """

        logging.debug(f"Fitting data cube with fit order={fit_order}.")
        self.data_cube.fit_cube(degree=fit_order)

    def make_flat_from_files(self, lo=10, hi=2000, calc_error=False, nsamples=None,
                             flat_lo=0.2, flat_hi=2., norm_to_dark=True):
        """
        Go through the files supplied to the module and generate a
        cube of rate images into an array. This method uses FlatDataCube
        class and its methods to generate flat rate images.

        Returns
        -------
        avg_rate_image: 2D array;
            The average of the rate_image_array in the z axis.
        lo: float; default 10,
            Minimum median (in a given sensor/exposure) count rate (in units of DN/s)
            for an image to be considered during the flat generation process.
        hi: float; default 2000,
            Maximum median (in a given sensor/exposure) count rate (in units of DN/s)
            for an image to be considered during the flat generation process.
        calc_error: bool,
            If `True` compute on the fly an uncertainty via bootstrap (slow).
            Default is `False`.
        nsamples: int; default None,
            Number of bootstrap samples to compute the uncertainty. If `None` we will sample
            half of the files used.
        flat_lo: float; default 0.2,
            If a flat value for a pixel is below `flat_lo` it is flagged and replaced by 1.
        flat_hi: float; default 2.0,
            If a flat value for a pixel is above `flat_hi` it is flagged and replaced by 1.
        norm_to_dark: bool; default True,
            If `True` normalize the flat to the dark element, i.e., use all image in an exposure
            and compute the median to obtain the normalization factor. If `False`, each detector
            is individually normalized.
        """

        logging.debug(
            "Making flat from the average flat rate of file list data cubes.")
        
        rate_image_array = None
 
        if nsamples is None:
            # Trying to avoid oversampling
            nsamples = max(1, int(self.num_files / 2))
        
        for fl in range(0, self.num_files):
            fname = self.file_list[fl]
            tmp = asdf.open(fname, lazy_tree=True)
            sca = int(tmp["roman"]["meta"]["instrument"]["detector"][-2:])
            t_start = tmp["roman"]["meta"]["exposure"]["start_time"]
            if len(np.shape(tmp.tree["roman"]["data"])) != 2:
                raise ValueError("The input data is expected to be a ramp (2D).\
                                  Please calculate or process your data first.")
            else:
                tmp_cube = tmp.tree["roman"]["data"].copy()
                if rate_image_array is None:
                    npixx, npixy = tmp_cube.shape
                    rate_image_array = np.full((self.num_files,
                                                 npixx,
                                                 npixy),
                                                 np.nan,
                                                 dtype=np.float32)
            tmp.close()
            if not isinstance(tmp_cube, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            # Only access data from quantity object.
            if isinstance(tmp_cube, u.Quantity):
                tmp_cube = tmp_cube.value
                logging.debug(
                    "Quantity object detected. Extracted data values.")

            # Sub-out infs by nans to ignore them safely
            tmp_cube[np.isinf(tmp_cube)] = np.nan
            # Check if the majority of pixels are not valid and skip if so
            num_nans = np.isnan(tmp_cube).sum()
            total_pixels = tmp_cube.size
            good_ratio = (total_pixels - num_nans) / total_pixels
            if good_ratio < 0.2:
                # If less than 20% of the pixels in an image are usable, skip that image
                logging.debug(f"Less than 20 percent of pixels usable. Skipped image {fname}")
            else:
                if norm_to_dark:
                    median = compute_fp_median(self.file_list[fl], npixx, npixy, t_start,
                                               sca, tmp_cube, self.file_map)
                else:
                    median = np.nanmedian(tmp_cube)
                # We will only consider images with median rates between "lo" and "hi"
                if lo <= median <= hi:
                    rate_image_array[fl, :, :] = tmp_cube/median  # Normalized L2
        
        # Compute "master" flat as median of individual flats pixel-by-pixel
        flat_image = np.nanmedian(rate_image_array, axis=0)
        self.flat_image = flat_image  # populate the attribute
        # Flag bad pixels
        self.update_data_quality_array(low_qe_threshold=flat_lo,
                                       flat_hi_threshold=flat_hi)
        # Force non-finite values to 1
        self.flat_image[~np.isfinite(self.flat_image)] = 1.0
        if calc_error:
            self.calculate_error(ind_flat_array=rate_image_array,
                                 nsamples=nsamples, nboot=nsamples)
            return self.flat_image, self.flat_error
        else:
            return self.flat_image

    def calculate_error(self, ind_flat_array=None,
                        nsamples=100, nboot=10, fill_random=False):
        """
        Calculate the uncertainty in the flat rate image using bootstrap resampling.
        If error array is None,
        generate random flat error array. If either nsamples or nboot are zero, just
        compute uncertainty as standard deviation of the resulting flats in a pixel.

        Parameters
        ----------
        ind_flat_array: ndarray; default = None,
           Array containing the individual ``flat" images for each exposure used to
           calculate the master flat.
        nsamples: int; default = 100,
           Number of samples for median bootstraping calculation.
        nboot: int; default = 10,
           Number of bootstrap samples.
        fill_random: bool; default = False,
           If `True` fill out the array with random numbers.
        """

        if fill_random:
            self.flat_error = np.random.randint(
                1, 11, size=(SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT)).astype(np.float32) / 100.
        else:
            if (nsamples > 0) & (nboot > 0):
                # We randomly select a subset of the images to calculate the median on them
                sel = np.random.choice(
                    np.arange(ind_flat_array.shape[0]), size=(nsamples, nboot))
                median_samples = np.nanmedian(ind_flat_array[sel], axis=0)
                # Compute the standard deviation of the median estimates as the uncertainty
                flat_unc = np.nanstd(median_samples, axis=0)
                self.flat_error = flat_unc
            else:
                # If either nsamples <= 0 or nboot <= 0
                self.flat_error = np.nanstd(ind_flat_array, axis=0)

    def update_data_quality_array(self, low_qe_threshold=0.2,
                                  flat_hi_threshold=2.):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        flat_hi_threshold: float; default = 2.0,
           Limit above which to flag pixels as unreliable flat (too high).
        add_low_qe_pixels: bool; default = False,
        """

        logging.info(
            'Flagging unreliable flat pixels')
        # Flag bad pixels
        self.dq_mask[np.isnan(self.flat_image)
                  ] |= self.dqflag_defs['UNRELIABLE_FLAT'].value
        self.dq_mask[self.flat_image >
                  flat_hi_threshold] |= self.dqflag_defs['UNRELIABLE_FLAT'].value
        logging.info(
            'Flagging low quantum efficiency pixels and updating DQ array.')
        # Locate low qe pixel ni,nj positions in 2D array
        self.dq_mask[self.flat_image <
                  low_qe_threshold] |= self.dqflag_defs['LOW_QE'].value

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the flat field object from the data model.
        flat_datamodel_tree = FlatRefModel()
        flat_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        flat_datamodel_tree['data'] = self.flat_image.astype(np.float32)
        flat_datamodel_tree['err'] = self.flat_error.astype(np.float32)
        flat_datamodel_tree['dq'] = self.dq_mask

        return flat_datamodel_tree

    class FlatDataCube(DataCube):
        """
        FlatNoiseDataCube class derived from DataCube.
        Handles Flat specific cube calculations
        Provide common fitting methods to calculate cube properties, such as rate and intercept images, for reference types.

        Parameters
        -------
        self.ref_type_data: input data array in cube shape
        self.wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
        """

        def __init__(self, ref_type_data, wfi_type):
            # Inherit reference_type.
            super().__init__(
                data=ref_type_data,
                wfi_type=wfi_type,
            )
            # The linear slope coefficient of the fitted data cube.
            self.rate_image = None
            self.rate_image_err = None  # uncertainty in rate image
            self.intercept_image = None
            self.intercept_image_err = (
                None  # uncertainty in intercept image (could be variance?)
            )
            self.ramp_model = None  # Ramp model of data cube.
            self.coeffs_array = None  # Fitted coefficients to data cube.
            self.covars_array = None  # Fitted covariance array to data cube.

        def fit_cube(self, degree=1):
            """
            fit_cube will perform a linear least squares regression using np.polyfit of a certain
            pre-determined degree order polynomial. This method needs to be intentionally called to
            allow for pipeline inputs to easily be modified.

            Parameters
            -------
            degree: int, default=1
                Input order of polynomial to fit data cube. Degree = 1 is linear. Degree = 2 is quadratic.
            """

            logging.debug("Fitting data cube.")
            # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.

            try:
                self.coeffs_array, self.covars_array = np.polyfit(
                    self.time_array,
                    self.data.reshape(len(self.time_array), -1),
                    degree,
                    full=False,
                    cov=True,
                )
                # Reshape the parameter slope array into a 2D rate image.
                # TODO the reshape and indices here are for linear degree fit = 1 only; update to handle quadratic also
                self.rate_image = self.coeffs_array[0].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
                # Reshape the parameter y-intercept array into a 2D image.
                self.intercept_image = self.coeffs_array[1].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
            except (TypeError, ValueError) as e:
                logging.error(
                    f"Unable to initialize DarkDataCube with error {e}")
                # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error

        def make_ramp_model(self, order=1):
            """
            make_data_cube_model uses the calculated fitted coefficients from fit_cube() to create
            a linear (order=1) or quadratic (order=2) model to the input data cube.

            NOTE: The default behavior for fit_cube() and make_model() utilizes a linear fit to the input
            data cube of which a linear ramp model is created.

            Parameters
            -------
            order: int, default=1
               Order of model to the data cube. Degree = 1 is linear. Degree = 2 is quadratic.
            """

            logging.info("Making ramp model for the input read cube.")
            # Reshape the 2D array into a 1D array for input into np.polyfit().
            # The model fit parameters p and covariance matrix v are returned.
            try:
                # Reshape the returned covariance matrix slope fit error.
                # rate_var = v[0, 0, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO -VERIFY USE
                # returned covariance matrix intercept error.
                # intercept_var = v[1, 1, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO - VERIFY USE
                self.ramp_model = np.zeros(
                    (
                        self.num_reads,
                        self.num_i_pixels,
                        self.num_j_pixels,
                    ),
                    dtype=np.float32,
                )
                if order == 1:
                    # y = m * x + b
                    # where y is the pixel value for every read,
                    # m is the slope at that pixel or the rate image,
                    # x is time (this is the same value for every pixel in a read)
                    # b is the intercept value or intercept image.
                    for tt in range(0, len(self.time_array)):
                        self.ramp_model[tt, :, :] = (
                            self.rate_image * self.time_array[tt]
                            + self.intercept_image
                        )
                elif order == 2:
                    # y = ax^2 + bx + c
                    # where we dont have a single rate image anymore, we have coefficients
                    for tt in range(0, len(self.time_array)):
                        a, b, c = self.coeffs_array
                        self.ramp_model[tt, :, :] = (
                            a * self.time_array[tt] ** 2
                            + b * self.time_array[tt]
                            + c
                        )
                else:
                    raise ValueError(
                        "This function only supports polynomials of order 1 or 2."
                    )
            except (ValueError, TypeError) as e:
                logging.error(f"Unable to make_ramp_cube_model with error {e}")
                # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error


def _process_single_sca(ifp, sca, fname, tmp_cube, t_start, resolved_file_map=None):
    """
    Worker function to process a single SCA index.
    
    Parameters
    ----------
    resolved_file_map : dict, optional
        Pre-calculated dictionary mapping (ifp, original_timestamp) -> exact_file_path.
        Crucial for avoiding slow dynamic globbing inside parallel processes.
    """
    if ifp == sca:
        return ifp - 1, tmp_cube

    # 1. Resolve filename using a pre-mapped dictionary if available
    if 'WFI' in fname:
        fname_aux = fname.replace(f'WFI{sca:02d}', f'WFI{ifp:02d}')
    elif 'wfi' in fname:
        fname_aux = fname.replace(f'wfi{sca:02d}', f'wfi{ifp:02d}')

    if not os.path.exists(fname_aux):
        if resolved_file_map and (ifp, fname) in resolved_file_map:
            fname_aux = resolved_file_map[(ifp, fname)]
        else:
            # Fallback path if mapping wasn't provided (slower)
            timestamp = fname_aux.split('/')[-1].split('_')[3]
            _files_here = glob.glob(fname_aux.replace(timestamp, '*'))
            for _f_h in _files_here:
                with asdf.open(_f_h, lazy_tree=True, lazy_load=True) as _ff:
                    t_here = _ff["roman"]["meta"]["exposure"]["start_time"]
                    # Calculate difference in total seconds quickly
                    if abs((t_here - t_start).sec) < 10:
                        fname_aux = _f_h
                        break

    # 2. Optimized Reading using Memory Mapping
    if os.path.exists(fname_aux):
        # Enable memory mapping explicitly to map array data directly into virtual memory
        with asdf.open(fname_aux, lazy_tree=True, memmap=True) as tmp:
            _data_ref = tmp["roman"]["data"]
            
            if isinstance(_data_ref, u.Quantity):
                # Inline extraction & explicit assignment conversion
                _data = np.asarray(_data_ref.value, dtype=np.float32)
            else:
                _data = np.asarray(_data_ref, dtype=np.float32)
                
            # Perform an in-place cleaning operation to save CPU cycles
            _data[np.isinf(_data)] = np.nan
            return ifp - 1, _data
            
    else:
        # Avoid creating random structures; allocate cleanly filled array
        return ifp - 1, np.full(tmp_cube.shape, np.nan, dtype=np.float32)
                


def compute_fp_median(fname, npixx, npixy, t_start, sca, tmp_cube, resolved_file_map):
    # Note: Added tmp_cube to the arguments as it was missing in the original signature
    tmp_fp = np.zeros((18, npixx, npixy), dtype=np.float32)
    
    # Use ProcessPoolExecutor for CPU-bound/IO-bound ASDF parsing
    # Automatically scales to available CPU cores
    with ProcessPoolExecutor() as executor:
        # Submit jobs for all 18 SCAs
        futures = [
            executor.submit(_process_single_sca, ifp, sca, fname, tmp_cube, t_start, resolved_file_map)
            for ifp in range(1, 19)
        ]
        
        # Collect results as they finish
        for future in futures:
            idx, data_array = future.result()
            tmp_fp[idx, :, :] = data_array

    median = np.nanmedian(tmp_fp)
    return median

