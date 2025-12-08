import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u

from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark
from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT

from ..reference_type import ReferenceType


class Dark(ReferenceType):
    """
    Class Dark() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    Dark() creates the dark reference file using roman data models
    and has all necessary meta and matching criteria for delivery to CRDS.

    Under automated operations conditions, a super dark file will be opened
    to fit a linear rate to the superdark cube.
    The dark rate image slope is derived from the super dark cube.

    NOTE: RFP Development Strategy is for explicit method calls to run code to populate reference type data models

    Example file creation commands:
    With user cube and even spacing.
    dark_obj = Dark(meta_data, ref_type_data=input_data_cube)
    dark_obj.make_rate_image_from_data_cube()
    dark_obj.calculate_error()
    dark_obj.update_data_quality_array()
    dark_bj.generate_outfile()

    With file list is superdark.asdf and even spacing.
    dark_obj = Dark(meta_data, file_list=superdark.asdf)
    dark_obj.calculate_error()
    dark_obj.update_data_quality_array()
    dark_bj.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_dark.asdf",
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
        outfile: string; default = roman_dark.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

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
        if not isinstance(meta_data, WFIMetaDark):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDark"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI dark reference file."

        logging.debug(f"Default dark reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.dark_rate_image = None
        self.dark_rate_image_error = None

        # TODO get from Dark config yml
        # Load default parameters from config_dark.yml
        self.hot_pixel_rate = 0
        self.warm_pixel_rate = 0
        self.dead_pixel_rate = 0

        # Module flow for creating reference file.
        # The fle list should only be one file, the super dark file.
        if self.file_list:
            # Expect exactly one super dark file.
            if len(self.file_list) > 1:
                raise ValueError(
                    f"A single super dark was expected in file_list. "
                    f"Received: {self.file_list}"
                )

            # Extract data cube from the superdark file.
            self._get_data_cube_from_superdark_file()

            logging.info(
                "Must call make_rate_image_from_data_cube and "
            )
        else:
            # Validate input data type.
            if not isinstance(ref_type_data, np.ndarray):
                raise TypeError("Input data must be a numpy array.")
            
            dim = ref_type_data.shape
            if len(dim) == 3:
                logging.info("User supplied 3D data cube to make dark reference file.")
                self.data_cube = self.DarkDataCube(ref_type_data, self.meta_data.type)
                logging.info("Must call make_rate_image_from_data_cube to get rate image.")
            elif len(dim) == 2:
                logging.info("User supplied 2D data array assumed to be dark rate.")
                self.dark_rate_image = ref_type_data
            else:
                raise ValueError("Input data must be a 2D or 3D numpy array.")

    def _get_data_cube_from_superdark_file(self):
        """
        Method to open superdark asdf file and get data.
        """

        logging.info(
            "OPENING - " + self.file_list[0]
        )  # Already checked that file_list is of length one.
        af = asdf.open(self.file_list[0]) # Use asdf open here for formatting
        data = af.tree['roman']['data']
        if isinstance(data, u.Quantity):  # Only access data from quantity object.
            data = data.value
        self.data_cube = self.DarkDataCube(data, self.meta_data.type)

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
        self.dark_rate_image = self.data_cube.rate_image
        self.dark_rate_image_error = self.data_cube.rate_image_err

    def calculate_error(self):
        """
        Populate dark rate error array. Create a uniform error array where each pixel
        is set using the detector-specific sigma value from the Table 1 of Betti et al. 2025
        'The Statistical Properties of Dark Ramps for the Roman-WFI Detectors'.
        """

        # Uncertainity values from the Table 1, in DN/s, from Betti et al. 2025
        DETECTOR_RATE_UNCERTAINTY = {
            "WFI01": 0.0059,
            "WFI02": 0.0035,
            "WFI03": 0.0065,
            "WFI04": 0.0046,
            "WFI05": 0.0049,
            "WFI06": 0.0048,
            "WFI07": 0.0052,
            "WFI08": 0.0041,
            "WFI09": 0.0035,
            "WFI10": 0.0040,
            "WFI11": 0.0060,
            "WFI12": 0.0048,
            "WFI13": 0.0043,
            "WFI14": 0.0044,
            "WFI15": 0.0034,
            "WFI16": 0.0039,
            "WFI17": 0.0049,
            "WFI18": 0.0052,
        }

        detector = self.meta_data.instrument_detector

        # Validate detector exists
        try:
            sigma = DETECTOR_RATE_UNCERTAINTY[detector]
        except KeyError:
            raise KeyError(
                f"Detector '{detector}' not found in DETECTOR_RATE_UNCERTAINTY "
                f"(expected keys: {list(DETECTOR_RATE_UNCERTAINTY.keys())})."
            )

        # Look up value from table
        sigma = DETECTOR_RATE_UNCERTAINTY[detector]
        self.dark_rate_image_error = np.full(
            (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT),
            sigma,
            dtype=np.float32
        )

        # Update for reference pixel border which has no dark rate and no dark rate error
        self.dark_rate_image_error[:4, :] = 0
        self.dark_rate_image_error[-4:, :] = 0
        self.dark_rate_image_error[:, :4] = 0
        self.dark_rate_image_error[:, -4:] = 0

    def update_data_quality_array(self, hot_pixel_rate, warm_pixel_rate):
        """
        The hot and warm pixel thresholds are applied to the dark_rate_image and the pixels are identified with their respective
        DQ bit flag.

        Parameters
        ----------
        hot_pixel_rate: float; default = 0.25 DN/s
            Determined from TVAC data by S Gomez
        warm_pixel_rate: float; default = 0.050 e/s
            Determined from TVAC data by S Gomez
        """

        self.hot_pixel_rate = hot_pixel_rate
        self.warm_pixel_rate = warm_pixel_rate

        logging.info("Flagging hot and warm pixels and updating DQ array.")
        # Locate hot and warm pixel num_i_pixels, num_j_pixels positions in 2D array
        self.dq_mask[self.dark_rate_image >= self.hot_pixel_rate] += self.dqflag_defs["HOT"]
        self.dq_mask[(self.dark_rate_image >= self.warm_pixel_rate)
                  & (self.dark_rate_image < self.hot_pixel_rate)] += self.dqflag_defs["WARM"]

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        dark_datamodel_tree = rds.DarkRef()
        dark_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        dark_datamodel_tree["dark_slope"] = self.dark_rate_image.astype(np.float32)
        dark_datamodel_tree["dark_slope_error"] = self.dark_rate_image_error.astype(np.float32)
        dark_datamodel_tree["dq"] = self.dq_mask

        return dark_datamodel_tree

    class DarkDataCube(DataCube):
        """
        DarkDataCube class derived from DataCube.
        Handles Dark specific cube information
        Provide common fitting methods to calculate cube properties.

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
                #TODO the reshape and indices here are for linear degree fit = 1 only; update to handle quadratic also
                self.rate_image = self.coeffs_array[0].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
                self.rate_image_err = (
                    self.covars_array[0, 0, :]
                    .reshape(self.num_i_pixels, self.num_j_pixels)
                    .astype(np.float32)
                )  # covariance matrix slope variance

                # Reshape the parameter y-intercept array into a 2D image.
                self.intercept_image = self.coeffs_array[1].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
                self.intercept_image_err = self.covars_array[1, 1, :].reshape(
                    self.num_i_pixels, self.num_j_pixels
                )
            except (TypeError, ValueError) as e:
                logging.error(f"Unable to initialize DarkDataCube with error {e}")
                # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error

        def make_ramp_model(self, order=1):
            """
            make_ramp_model uses the calculated fitted coefficients from fit_cube() to create
            a linear (order=1) or quadratic (order=2) model of the input data cube.

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
