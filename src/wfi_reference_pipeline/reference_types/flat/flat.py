import logging

import asdf
import numpy as np
import roman_datamodels as rdm
import roman_datamodels.stnode as rds
from astropy import units as u
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat

from ..reference_type import ReferenceType


class Flat(ReferenceType):

    """
    Class Flat() inherits the ReferenceType() base class methods where
    static meta data for all reference file types are written. The class
    ingests a list of files and finds all exposures with the same filter
    within some maximum date range. Fit ramps to all available filter
    cubes ro generate flat rate images and average together and normalize
    to produce the filter dependent flat rate image.
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

        # Input dimensions of science array for ReferenceType() to
        # to properly generate dq array mask for Flat().
        #if bit_mask is None:
            #bit_mask = np.zeros((4088, 4088), dtype=np.uint32)

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber,
            make_mask=True,
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaFlat):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaFlat"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI dark reference file."

        logging.debug(f"Default flat reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.flat_image = None  # The attribute 'data' in data model.
        self.flat_error = None  # The attribute assigned to the flat['err'].
        self.num_files = 0

        # # Inputs
        # self.input_read_cube = input_flat_cube  # Supplied input read cube.
        # # Internal
        # self.rate_array = None  # 2D rate array of the science pixels.
        # self.rate_var_array = None  # 3D cute of rate arrays.
        # # Flattened attributes
        # self.flat_rate_image = None  # The attribute assigned to the flat['data'].
        # self.flat_rate_image_var = None  # Variance in fitted rate image.
        # self.flat_intercept = None  # Intercept image from ramp fit.
        # self.flat_intercept_var = None  # Variance in fitted intercept image.

        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.num_files = len(self.file_list)
            # Must make_flat_rate_image() to finish creating reference file.
        else:
            if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            if isinstance(
                ref_type_data, u.Quantity
            ):  # Only access data from quantity object.
                ref_type_data = ref_type_data.value
                logging.debug("Quantity object detected. Extracted data values.")

            dim = ref_type_data.shape
            if len(dim) == 2:
                logging.debug("The input 2D data array is now self.flat_image.")
                self.flat_image = ref_type_data
                logging.debug("Ready to generate reference file.")
            elif len(dim) == 3:
                logging.debug(
                    "User supplied 3D data cube to make flat reference file."
                )
                self.data_cube = self.FlatDataCube(
                    ref_type_data, self.meta_data.type
                )

                # Must call make_flat_rate_image() to finish creating reference file.
                logging.debug(
                    "Must call make_flat_rate_image() to finish creating reference file."
                )
            else:
                raise ValueError(
                    "Input data is not a valid numpy array of dimension 2 or 3."
                )

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

    def make_flat_from_files(self):
        """
        Go through the files supplied to the module and generate a
        cube of rate images into an array. This method uses the
        make_flat_from_cube method also so that the number of reads
        can vary over multiple input read cubes. The average of all
        rate arrays in the cube are averaged and returned.
        return:

        Returns
        -------
        avg_rate_image: 2D array;
            The average of the rate_image_array in the z axis.
        """

        print("Inside make_flat_from_files() method.")
        n_reads_per_fl_arr = np.zeros(self.num_files)
        rate_image_array = np.zeros((self.num_files,
                                     self.data_cube.num_i_pixels,
                                     self.data_cube.num_j_pixels),
                                    dtype=np.float32)
        rate_image_var_array = np.zeros((self.num_files,
                                         self.data_cube.num_i_pixels,
                                         self.data_cube.num_j_pixels),
                                        dtype=np.float32)
        for fl in range(0, self.num_files):
            tmp = asdf.open(self.file_list[fl])
            n_reads_per_fl_arr[fl], _, _ = np.shape(tmp.tree["roman"]["data"])
            tmp_cube = tmp.tree["roman"]["data"]
            if not isinstance(tmp_cube, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            if isinstance(
                tmp_cube, u.Quantity
            ):  # Only access data from quantity object.
                tmp_cube = tmp_cube.value
                logging.debug("Quantity object detected. Extracted data values.")
            self.data_cube = self.FlatDataCube(tmp_cube, self.meta_data.type)
            self.data_cube.fit_cube(degree=1)
            rate_image_array[fl, :, :] = self.data_cube.rate_image
            rate_image_var_array[fl, :, :] = self.data_cube.covars_array
            tmp.close()

        avg_rate_image = np.mean(rate_image_array, axis=0)
        return avg_rate_image

    # def make_flat_from_cube(self, n_reads, ni=None):
    #     """
    #     Method finds the fitted rate and variance by first initialize arrays
    #     by the number of reads and the number of pixels. The fitted ramp or slope
    #     along the time axis for the number of reads in the cube using a 1st order
    #     polyfit. The best fit solutions and variance are returned.
    #
    #     Parameters
    #     ----------
    #     n_reads: integer; Positional required.
    #         Number of reads to initialize fitted arrays.
    #     ni: integer; Default: None.
    #         Number of reads to initialize.
    #
    #     Returns
    #     -------
    #     rate_image: 2D array;
    #         The fitted rate image from the cube.
    #     rate_image_var: 2D array;
    #         The variance of the fitted rate image from the cube.
    #     """
    #
    #     # If ni is supplied, overwrite attribute.
    #     if ni is not None:
    #         self.ni = ni
    #
    #     # Make the time array for the length of the dark read cube exposure.
    #     if self.meta['instrument']['optical_element']:
    #         self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
    #     else:
    #         raise ValueError('Optical element not found; this might not be a flat file.')
    #     time_array = np.array(
    #         [self.frame_time * i for i in range(1, n_reads + 1)]
    #     )
    #
    #     p, c = np.polyfit(time_array,
    #                       self.input_read_cube.reshape(len(time_array), -1), 1, full=False, cov=True)
    #
    #     # Reshape results back to 2D arrays.
    #     rate_image = p[0].reshape(self.ni, self.ni).astype(np.float32)  # the fitted ramp slope image
    #     rate_image_var = c[0, 0, :].reshape(self.ni, self.ni).astype(np.float32)  # covariance matrix slope variance
    #
    #     return rate_image, rate_image_var

    def calculate_error(self):
        """
        Calculate the uncertainty in the flat rate image.
        """
        # TODO for future implementation
        # high_flux_err = 1.2 * self.flat_rate_image * (n_reads * 2 + 1) /
        # (n_reads * (n_reads * 2 - 1) * self.frame_time)
        #

        self.flat_error = np.random.randint(1, 11, size=(4088, 4088)).astype(np.float32) / 100.

    def update_data_quality_array(self, low_qe_threshold=0.2):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        """

        # TODO remove random loq qe pixels from flat_rate_image
        # Generate between 200-300 pixels with low qe for DMS builds
        rand_num_lowqe = np.random.randint(200, 300)
        coords_x = np.random.randint(0, 4088, rand_num_lowqe)
        coords_y = np.random.randint(0, 4088, rand_num_lowqe)
        rand_low_qe_values = np.random.randint(5, 20, rand_num_lowqe) / 100.  # low eq in range 0.05 - 0.2
        self.flat_image[coords_x, coords_y] = rand_low_qe_values

        self.low_qe_threshold = low_qe_threshold

        logging.info('Flagging low quantum efficiency pixels and updating DQ array.')
        # Locate low qe pixel ni,nj positions in 2D array
        self.mask[self.flat_image < self.low_qe_threshold] += self.dqflag_defs['LOW_QE']

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the flat field object from the data model.
        flat_datamodel_tree = rds.FlatRef()
        flat_datamodel_tree['meta'] = self.meta_data
        flat_datamodel_tree['data'] = self.flat_image
        flat_datamodel_tree['dq'] = self.mask
        flat_datamodel_tree['err'] = self.flat_error

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
            self.rate_image = None  # The linear slope coefficient of the fitted data cube.
            self.intercept_image = (
                None  # the y intercept of a line fit to the data_cube
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
                #TODO the reshape and indices here are for linear degree fit = 1 only; update to handle quadratic also
                self.rate_image = self.coeffs_array[0].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
                # Reshape the parameter y-intercept array into a 2D image.
                self.intercept_image = self.coeffs_array[1].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
            except (TypeError, ValueError) as e:
                logging.error(f"Unable to initialize DarkDataCube with error {e}")
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
