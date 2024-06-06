import datetime
import logging

import numpy as np
import roman_datamodels as rdm
import roman_datamodels.stnode as rds
from astropy import units as u
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark

from ..reference_type import ReferenceType


class Dark(ReferenceType):
    """
    Class Dark() inherits the ReferenceType() base class methods where static meta data for all reference
    file types are written. Under automated operations conditions, a list of dark calibration files from a directory
    will be the input data for the class to begin generating a dark reference file. A super dark cube of all
    available reads and exposures will go through outlier rejection and then averaging per a multi-accumulation (MA)
    prescription that is retrieved from the RTB database. The resampled MA-table dark cube is analyzed further to
    flag hot and warm pixels. Statistics on these or additional quantities can be written to the RTB database for
    detector performance monitoring with time and comparison across WFI. The dark reference file created is then
    written to disk.

    NOTE: RFP Development Strategy is for explicit method calls to run code to populate reference type data models

    #TODO Example method calls using a superdark.asdf file
    dark = Dark(meta_data, file_list=superdark.asdf, ma_table_id=3)
    dark.make_dark_rate_image_from_data_cube()
    dark.make_ma_table_resampled_data()
    dark.calculate_error()
    dark.update_data_quality_array()
    dark.generate_outfile()

    #TODO Example method calls with a user supplied data cube:
    dark = Dark(meta_data, ref_type_data=user_cube, ...)
    dark.make_dark_rate_image_from_data_cube()
    dark.make_ma_table_resampled_data(None, None, user_read_pattern) for uneven spacing using list of lists
        OR
        dark.make_ma_table_resampled_data(num_resultants, num_reads_per_resultant) for even spacing
    dark.calculate_error()
    dark.update_data_quality_array()
    dark.generate_outfile()
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
        outfile: string; default = roman_readnoise.asdf
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
            clobber=clobber,
            make_mask=True,
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaDark):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDark"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI dark reference file."

        logging.debug(f"Default dark reference file object: {outfile} ")

        # This SHOULD only be one file in the file list, and it is the SuperDark file
        if self.file_list:
            # Get file list properties and select data cube.
            if len(self.file_list) > 1:
                raise ValueError("A single super dark was expected in file_list..")
            else:
                self.get_data_cube_from_superdark_file()
            # Must make_ma_table_resampled_cube and then make_dark_rate_image()
        else:
            if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            if isinstance(
                ref_type_data, u.Quantity
            ):  # Only access data from quantity object.
                ref_type_data = ref_type_data.value
                logging.info("Quantity object detected. Extracted data values.")
            dim = ref_type_data.shape
            if len(dim) == 3:
                logging.info("User supplied 3D data cube to make dark reference file.")
                self.data_cube = self.DarkDataCube(ref_type_data, self.meta_data.type)
                # Must make_ma_table_resampled_cube and then make_dark_rate_image()
                logging.info(
                    "Must call make_ma_table_resampled_cube and then make_dark_rate_image() to "
                    "finish creating reference file."
                )
            else:
                raise ValueError(
                    "Input data is not a valid numpy array of dimension 3."
                )

        # MA Table attributes
        # TODO populate from MA Table Config file
        self.ma_table_read_pattern = None  # read pattern from ma table meta data -
        # nested list of lists reads in resultants will be replacing (ngroups, nframes, groupgap)
        self.num_resultants = None

        # Attributes for initialized arrays for resampling.
        self.resampled_data = None
        self.resampled_data_err = None
        self.resampled_model = None
        self.resultant_tau_array = None

        # TODO get from Dark config yml
        # Load default parameters from config_dark.yml
        self.hot_pixel_rate = 0
        self.warm_pixel_rate = 0
        self.dead_pixel_rate = 0

        # NOTE: keeping the below just for reference when setting up, can delete if Rick doesnt need these
        # self.intercept_image = np.zeros((self.num_i_pixels, self.num_j_pixels), dtype=np.float32) # Intercept image from ramp fit.
        # self.intercept_var = np.zeros((self.num_i_pixels, self.num_j_pixels), dtype=np.float32) # Variance in fitted intercept image.

    def get_data_cube_from_superdark_file(self):
        """
        Method to open superdark asdf file and get data.
        """

        logging.info(
            "OPENING - " + self.file_list
        )  # Already checked that file_list is of length one.
        data = rdm.open(self.file_list)
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

    def make_ma_table_resampled_data(self,
                                     num_resultants=None,
                                     num_reads_per_resultant=None,
                                     read_pattern=None):
        """
        The method make_ma_table_resampled_cube() uses the input read_pattern, which is a nested list of lists,
        or the number of resultants and reads per resultant to average reads into resultants. If read_pattern
        is supplied, the even spacing parameters will be ignored.

        Parameters
        ----------
        num_resultants: integer; Default=None
            The number of resultants.
        num_reads_per_resultant: integer; Default=None
            The user supplied number of reads per resultant in evenly spaced resultants.
        read_pattern: list of lists; Default=None
            Nested list of lists with integers for averaging reads into resultants.
        """

        if read_pattern:
            self.num_resultants = len(read_pattern)
        else:
            self.num_resultants = num_resultants

        self.resampled_data = np.zeros((self.num_resultants,
                                        self.data_cube.num_i_pixels,
                                        self.data_cube.num_j_pixels), dtype=np.float32)
        self.resultant_tau_array = np.zeros(self.num_resultants, dtype=np.float32)

        if read_pattern:
            # Use read pattern for resampling by averaging reads into resultants and
            # get mean time of resultant for tau array
            # Iterate over each nested list in the read pattern
            logging.debug("Averaging over reads following read pattern supplied.")
            for resultant_i, read_pattern_frames in enumerate(read_pattern):
                # Get the average time for the list of frames in the read pattern
                read_pattern_zero_indices = [
                    i - 1 for i in read_pattern_frames
                ]  # zero index for time array
                self.resultant_tau_array[resultant_i] = np.mean(
                    self.data_cube.time_array[read_pattern_zero_indices]
                )  # TODO - do we need this? DMS calculates this for us
                # Average the data by summing read by read and dividing by number of raeds
                for read_i in read_pattern_frames:
                    self.resampled_data[resultant_i] += self.data_cube.data[
                        read_i - 1
                    ]  # Adjusted for 0 indexing
                self.resampled_data[resultant_i] /= len(read_pattern_frames)
            logging.debug("Finished re-sampling with read pattern.")
        else:
            # Use even spacing resultant and reads per resultant provided to the method and
            # get mean time of resultant for tau array.
            if not isinstance(num_resultants, int) or not isinstance(num_reads_per_resultant, int):
                raise ValueError(
                    "Both num_resultants and num_reads_per_resultant must be integers."
                )
            if num_resultants is None or num_reads_per_resultant is None:
                raise ValueError(
                    "Both num_resultants and num_reads_per_resultant are required inputs for"
                    "MA Table resampling with even spacing."
                )
            logging.debug("Averaging over reads with evenly spaced resultants.")
            if num_reads_per_resultant > self.data_cube.num_reads:
                raise ValueError(
                    "Cannot average over reads greater in length than data."
                )
            # Averaging with even spacing.
            for resultant_i in range(self.num_resultants):
                i1 = resultant_i * num_reads_per_resultant
                i2 = i1 + num_reads_per_resultant
                if i2 > self.data_cube.num_reads:
                    logging.warning(
                        "Warning: The number of reads per resultant was not evenly divisible into the number"
                        " of available reads to average and remainder reads were skipped."
                    )
                    logging.warning(
                        f"Resultants after resultant {resultant_i+1} contain zeros."
                    )
                    break  # Remaining reads cannot be evenly divided
                self.resampled_data[resultant_i, :, :] = np.mean(
                    self.data_cube.data[i1:i2, :, :], axis=0
                )
                self.resultant_tau_array[resultant_i] = np.mean(
                    self.data_cube.time_array[i1:i2]
                )

            logging.info(
                f"MA Table resampling with {self.num_resultants} resultants averaging {num_reads_per_resultant}"
                f" reads per resultant complete."
            )

    def make_metrics_dicts(self):
        """
        The method make_metrics_dicts is used to create reference file type specific
        metrics for tracking by the data monitoring tool from entries into the
        the RTB Database.
        """

        # Create the dark file dictionary
        db_dark_fl_dict = {
            "detector": self.meta_data["instrument"]["detector"],
            "exposure_type": self.meta_data["exposure"]["type"],
            "created_date": datetime.datetime.utcnow(),
            "use_after": datetime.datetime.utcnow(),
            "crds_filename": "test_crds_filename.asdf",
            "crds_delivery_id": 1,
        }

        hot_pixel_mask = self.data_cube.rate_image > self.hot_pixel_rate
        num_hot_pixels = np.sum(hot_pixel_mask)
        warm_pixel_mask = (self.warm_pixel_rate <= self.data_cube.rate_image) & (
            self.data_cube.rate_image < self.hot_pixel_rate
        )
        num_warm_pixels = np.sum(warm_pixel_mask)
        dead_pixel_mask = self.data_cube.rate_image < self.dead_pixel_rate
        num_dead_pixels = np.sum(dead_pixel_mask)

        logging.info(
            f"Found {num_hot_pixels} hot pixels,  {num_warm_pixels} warm pixels, and {num_dead_pixels} were"
            f"found in the dark rate ramp image."
        )
        print(
            "hot, warm, dead pixels", num_hot_pixels, num_warm_pixels, num_dead_pixels
        )

        # Create the dark dq dictionary
        db_dark_dq_dict = {
            "num_dead_pix": num_dead_pixels,
            "num_hot_pix": num_hot_pixels,
            "num_warm_pix": num_warm_pixels,
            "hot_pix_rate": 0.015,
            "warm_pix_rate": 0.010,
        }

        # Number of SCA amplifiers (4096 pixels / 128 pixels)
        amp_pixel_width = 128
        num_amps = 32
        # Initialize empty lists to store median and mean values for each amplifier
        median_values = []
        mean_values = []

        # Loop through amplifiers and find stats by amplifier
        for i in range(num_amps):
            start_index = i * amp_pixel_width
            end_index = (i + 1) * amp_pixel_width
            amp_i = self.data_cube.rate_image[:, start_index:end_index]
            median = np.nanmedian(amp_i)
            mean = np.nanmean(amp_i)

            # Append the results to the lists
            median_values.append(median)
            mean_values.append(mean)

        # Create the dark amp dictionary
        db_dark_amp_dict = {
            "amp_id": list(range(1, num_amps + 1)),
            "median_dark_current": median_values,
            "mean_dark_current": mean_values,
        }

        # Create the dark structure dictionary
        db_dark_struc_dict = {"coefficient": 42.0}

        # TODO - Verify Use of metric_dict
        # metric_dict = {
        #     'db_dark_fl_dict': db_dark_fl_dict,
        #     'db_dark_dq_dict': db_dark_dq_dict,
        #     'db_dark_struc_dict': db_dark_struc_dict,
        #     'db_dark_amp_dict': db_dark_amp_dict
        # }

        return db_dark_fl_dict, db_dark_dq_dict, db_dark_struc_dict, db_dark_amp_dict

    def calculate_error(self):
        """
        #TODO re-evaluate the error propagation for dark and how fit_dark_ramp() and this method operate and computational performance
        Old method of computing dark error
        """

        self.resampled_data_err = np.zeros((self.num_resultants,
                                            self.data_cube.num_i_pixels,
                                            self.data_cube.num_j_pixels), dtype=np.float32)

        # Generate a dark ramp cube model per the resampled ma table specs.
        self.resampled_model = np.zeros(
            (
                len(self.resampled_data),
                self.data_cube.num_i_pixels,
                self.data_cube.num_j_pixels,
            ),
            dtype=np.float32,
        )
        for tt in range(0, len(self.resultant_tau_array)):
            self.resampled_model[tt, :, :] = (
                self.data_cube.rate_image * self.resultant_tau_array[tt]
                + self.data_cube.intercept_image
            )  # y = m*x + b
        # Calculate the residuals of the dark ramp model and the data
        residual_cube = self.resampled_model - self.resampled_data
        std = np.std(residual_cube, axis=0)
        # This is the standard deviation of residuals from the resampled cube
        # model and the resampled cube data. Therefore std^2 is the resampled read noise variance.
        # The dark cube error array should be a 2D image of 4096x4096 with the slope variance from the model fit
        # and the variance of the resampled residuals are added in quadrature.
        self.resampled_data_err[0, :, :] = (
            std * std + self.data_cube.rate_image_err
        ) ** 0.5

    def update_data_quality_array(self, hot_pixel_rate=0.015, warm_pixel_rate=0.010, dead_pixel_rate=0.0001):
        # TODO evaluate options for variabiles like this and sigma clipping with a parameter file?
        """
        The hot and warm pixel thresholds are applied to the dark_rate_image and the pixels are identified with their respective
        DQ bit flag.

        Parameters
        ----------
        dead_pixel_rate: float; default = 0.0001 DN/s or ADU/s
            The dead pixel rate is the number of DN/s determined from detector characterization to be the level at
            which no detectable signal from dark current would be found in a very long exposure.
        hot_pixel_rate: float; default = 0.015 DN/s or ADU/s
            The hot pixel rate is the number of DN/s determined from detector characterization to be 10-sigma above
            the nominal expectation of dark current.
        warm_pixel_rate: float; default = 0.010 e/s
            The warm pixel rate is the number of DN/s determined from detector characterization to be 8-sigma above
            the nominal expectation of dark current.
        """

        self.hot_pixel_rate = hot_pixel_rate
        self.warm_pixel_rate = warm_pixel_rate
        self.dead_pixel_rate = dead_pixel_rate

        logging.info("Flagging dead, hot, and warm pixels and updating DQ array.")
        # Locate hot and warm pixel num_i_pixels, num_j_pixels positions in 2D array
        self.mask[self.data_cube.rate_image > self.hot_pixel_rate] += self.dqflag_defs[
            "HOT"
        ]
        self.mask[
            (self.warm_pixel_rate <= self.data_cube.rate_image)
            & (self.data_cube.rate_image < self.hot_pixel_rate)
        ] += self.dqflag_defs["WARM"]
        self.mask[self.data_cube.rate_image < self.dead_pixel_rate] += self.dqflag_defs[
            "DEAD"
        ]

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        dark_datamodel_tree = rds.DarkRef()
        dark_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        dark_datamodel_tree["data"] = self.resampled_data * u.DN
        dark_datamodel_tree["dark_slope"] = self.data_cube.rate_image.astype(np.float32) * u.DN / u.s
        dark_datamodel_tree["dark_slope_error"] = (
            (self.data_cube.rate_image_err**0.5) * u.DN / u.s
        )
        dark_datamodel_tree["dq"] = self.mask

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
