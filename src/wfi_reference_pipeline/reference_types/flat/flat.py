import logging
import asdf
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat
from wfi_reference_pipeline.constants import (
    WFI_TYPE_IMAGE,
)

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
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Default bit mask size of 4088x4088 for flat is size of science array
        # and must be provided if not bit_mask to instantiate properly in base class.
        if bit_mask is None:
            bit_mask = np.zeros((4088, 4088), dtype=np.uint32)

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
                self.flat_error = np.zeros((4088, 4088), dtype=np.float32)
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
            dq = self.mask
        Additional method calls must be run to populate initialized arrays:
            self.calculate_error()
            self.update_data_quality_array()
        """

        if self.file_list:
            logging.debug(
                "Making flat_image from average of rate images from file list."
            )
            avg_rate_image = self.make_flat_from_files()
            self.flat_image = avg_rate_image / np.mean(avg_rate_image)
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
        self.flat_error = np.zeros((4088, 4088), dtype=np.float32)
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

    def make_flat_from_files(self, lo=100, hi=500, calc_error=False, nsamples=100,
                             flat_lo=1e-3, flat_hi=2.):
        """
        Go through the files supplied to the module and generate a
        cube of rate images into an array. This method uses FlatDataCube
        class and its methods to generate flat rate images.

        Returns
        -------
        avg_rate_image: 2D array;
            The average of the rate_image_array in the z axis.
        lo: float;
            Minimum median (in a given sensor/exposure) count rate (in units of DN/s)
            for an image to be considered during the flat generation process.
        hi: float;
            Maximum median (in a given sensor/exposure) count rate (in units of DN/s)
            for an image to be considered during the flat generation process.
        calc_error: bool,
            If `True` compute on the fly an uncertainty via bootstrap (slow).
            Default is `False`.
        nsamples: int;
            Number of bootstrap samples to compute the uncertainty.
        flat_lo: float;
            If a flat value for a pixel is below `flat_lo` it is flagged and replaced by 1.
        flat_hi: float;
            If a flat value for a pixel is above `flat_hi` it is flagged and replaced by 1.
        """

        logging.debug(
            "Making flat from the average flat rate of file list data cubes.")

        for fl in range(0, self.num_files):
            print('Working with', self.file_list[fl])
            tmp = asdf.open(self.file_list[fl])
            npixx, npixy = np.shape(tmp.tree["roman"]["data"])
            tmp_cube = tmp.tree["roman"]["data"].copy()
            if fl == 0:
                rate_image_array = np.zeros((self.num_files,
                                             npixx,
                                             npixy),
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
            # We will normalize each L2 image by the median rate (ignoring infs/NaNs)
            median = np.nanmedian(tmp_cube)
            # We will only consider images with median rates between "lo" and "hi"
            if (median >= lo) & (median <= hi):
                rate_image_array[fl, :, :] = tmp_cube/median  # Normalized L2
            else:
                # This is a bit wasteful memory wise...
                rate_image_array[fl, :, :] = np.nan*np.ones_like(tmp_cube)

        flat_image = np.nanmedian(rate_image_array, axis=0)
        self.mask[np.isnan(flat_image)
                  ] += self.dqflag_defs['UNRELIABLE_FLAT'].value
        self.mask[flat_image <
                  flat_lo] += self.dqflag_defs['UNRELIABLE_FLAT'].value
        self.mask[flat_image >
                  flat_hi] += self.dqflag_defs['UNRELIABLE_FLAT'].value
        if calc_error:
            # We randomly select a subset of the images to calculate the median on them
            sel = np.random.randint((nsamples, self.num_files/2))
            median_samples = np.nanmedian(rate_image_array[sel], axis=1)
            # Compute the standard deviation of the median estimates as the uncertainty
            flat_unc = np.std(median_samples, axis=0)
            return flat_image, flat_unc
        else:
            return flat_image

    def calculate_error(self, error_array=None):
        """
        Calculate the uncertainty in the flat rate image. If error array is None,
        generate random flat error array.

        Parameters
        ----------
        error_array: ndarray; default = None,
           Variable to provide a precalculated error array. If None, random error is
           calculated for flat error array.
        """

        # TODO for future implementation from A. Petric
        # high_flux_err = 1.2 * self.flat_rate_image * (n_reads * 2 + 1) /
        # (n_reads * (n_reads * 2 - 1) * self.frame_time)
        #
        if error_array is None:
            self.flat_error = np.random.randint(
                1, 11, size=(4088, 4088)).astype(np.float32) / 100.
        else:
            self.flat_error = error_array

    def update_data_quality_array(self, low_qe_threshold=0.2, add_low_qe_pixels=False):
        """
        Update data quality array bit mask with flag integer value.

        If add_low_qr_pixels is True, a random number of loq quantum efficiency pixels
        will be inserted into self.flat_image.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        add_low_qe_pixels: bool; default = False,
        """

        if add_low_qe_pixels:
            # TODO remove random loq qe pixels from flat_rate_image
            # Generate between 200-300 pixels with low qe for DMS builds
            rand_num_lowqe = np.random.randint(200, 300)
            coords_x = np.random.randint(0, 4088, rand_num_lowqe)
            coords_y = np.random.randint(0, 4088, rand_num_lowqe)
            rand_low_qe_values = np.random.randint(
                5, 20, rand_num_lowqe) / 100.  # low eq in range 0.05 - 0.2
            self.flat_image[coords_x, coords_y] = rand_low_qe_values

        logging.info(
            'Flagging low quantum efficiency pixels and updating DQ array.')
        # Locate low qe pixel ni,nj positions in 2D array
        self.mask[self.flat_image <
                  low_qe_threshold] += self.dqflag_defs['LOW_QE']

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the flat field object from the data model.
        flat_datamodel_tree = rds.FlatRef()
        flat_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        flat_datamodel_tree['data'] = self.flat_image.astype(np.float32)
        flat_datamodel_tree['err'] = self.flat_error.astype(np.float32)
        flat_datamodel_tree['dq'] = self.mask

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
