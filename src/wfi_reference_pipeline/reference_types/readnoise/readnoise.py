import logging
import math
import os
import asdf
import crds
import subprocess
import datetime
from crds.client import api
from crds import getrecommendations
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.time import Time
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_readnoise import WFIMetaReadNoise

from ..reference_type import ReferenceType


class ReadNoise(ReferenceType):
    """
    Class ReadNoise() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    ReadNoise() creates the read noise reference file using roman data models
    and has all necessary meta and matching criteria for delivery to CRDS.

    Under automated operational conditions, a dark calibration file with the most number
    of reads for each detector will be selected from a file list of inputs. Dark calibration
    files where every read is available and not averaged are the best available data to
    measure the variance of the detector read by read. A ramp model for all available reads
    will be subtracted from the input data cube provided and the variance in the residuals
    is determined to be the best measurement of the read noise (Casertano and Cosentino email
    discussions Dec 2022).

    Additional complexity such as the treatment of Poisson noise, shot noise, read-out noise,
    etc. are to be determined. The method get_cds_noise() is available for diagnostics purposes
    and comparison when developing more mature functionality of the reference file pipeline.

    Example file creation commands:
    With user cube input.
    readnoise_obj = ReadNoise(meta_data, ref_type_data=input_data)
    readnoise_obj.make_readnoise_image()
    readnoise_obj.generate_outfile()

    From a file list with many cubes.
    readnoise_obj = ReadNoise(meta_data, file_list=input_file_list.txt)
    readnoise_obj._select_data_cube_from_file_list()
    readnoise_obj.make_readnoise_image()
    readnoise_obj.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_readnoise.asdf",
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
        outfile: string; default = roman_readnoise.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Inherit reference_type.
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaReadNoise):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaReadNoise"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI read noise reference file."

        logging.debug(f"Default read noise reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.readnoise_image = None  # The attribute 'data' in data model.
        self.ramp_res_var = None  # The variance of residuals from the difference of the ramp model and a data cube.
        self.cds_noise = None  # The correlated double sampling noise estimate between successive pairs
        self.num_files = 0

        self.residual_cube = None
        self.clipped_res_cube = None


        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.num_files = len(self.file_list)
            # Must make_readnoise_image() to finish creating reference file.
        else:
            if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            if isinstance(ref_type_data, u.Quantity):  # Only access data from quantity object.
                ref_type_data = ref_type_data.value
                logging.debug("Quantity object detected. Extracted data values.")

            dim = ref_type_data.shape
            if len(dim) == 2:
                logging.debug("The input 2D data array is now self.readnoise_image.")
                self.readnoise_image = ref_type_data
                logging.debug("Ready to generate reference file.")
            elif len(dim) == 3:
                logging.debug(
                    "User supplied 3D data cube to make read noise reference file."
                )
                self.data_cube = self.ReadNoiseDataCube(ref_type_data, self.meta_data.type)
                # Must call make_readnoise_image() to finish creating reference file.
                #TODO evaluate self.meta_data.type and exposure.type and exposure.p_exptype
                logging.debug(
                    "Must call make_readnoise_image() to finish creating reference file."
                )
            else:
                raise ValueError(
                    "Input data is not a valid numpy array of dimension 2 or 3."
                )

    def _select_data_cube_from_file_list(self):
        """
        Looks through the file list provided to ReadNoise() and finds the file with
        the most number of reads. It sorts the files in descending order by the number of reads such that the
        first index will be the file with the most number of reads and the last will have the fewest.
        Return the datacube with the longest number of reads.
        """

        logging.info(
            f"Using files from {os.path.dirname(self.file_list[0])} to find file longest exposure"
            f"and the most number of reads."
        )
        # Go through all files to sort them from the longest to shortest number of reads available.
        fl_reads_ordered_list = []
        for fl in range(0, self.num_files):
            # TODO update using rdm.open() method
            with asdf.open(self.file_list[fl]) as tmp:
                n_rds, _, _ = np.shape(tmp.tree["roman"]["data"])
                fl_reads_ordered_list.append([self.file_list[fl], n_rds])
        # Sort the list of files in reverse order such that the file with the most number of reads is always in
        # the zero index first element of the list.
        fl_reads_ordered_list.sort(key=lambda x: x[1], reverse=True)

        # Get the input file with the most number of reads from the sorted list.
        # TODO update using rdm.open() method
        with asdf.open(fl_reads_ordered_list[0][0]) as tmp:
            ref_type_data = tmp.tree["roman"]["data"]
            if isinstance(
                ref_type_data, u.Quantity
            ):  # Only access data from quantity object.
                ref_type_data = ref_type_data.value

        logging.debug(
            f"Using the file {fl_reads_ordered_list[0][0]} to get a read noise cube."
        )
        self.data_cube = self.ReadNoiseDataCube(ref_type_data, self.meta_data.type)

    def make_readnoise_image(self):
        """
        This method is used to generate the reference file image from the file list or a data cube.

        NOTE: This method is intended to be the module's internal pipeline where each method's internal
        variables and parameters are set and this is the single call to populate all attributes needed
        for the reference file data model.

        To use CDS noise, modify code as follows:
        self.readnoise_image = self.comp_cds_noise()
        """

        logging.info("Making read noise image.")
        if self.file_list:
            self._select_data_cube_from_file_list()
        self.make_rate_image_from_data_cube(fit_order=1)
        self.data_cube.make_ramp_model()
        print('Making read noise image')
        self.readnoise_image = self.comp_ramp_res_var()

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

    def comp_ramp_res_var(self, sig_clip_res_low=5.0, sig_clip_res_high=5.0):
        """
        Compute the variance of the residuals to a ramp fit. The method get_ramp_res_var() finds the difference between
        the fitted ramp model and the input read cube  provided and calculates the variance of the residuals. This is
        the most appropriate estimation for the read noise for WFI (Casterano and Cosentino email discussions Dec 2022).

        Parameters
        ----------
        sig_clip_res_low: float; default = 5.0
            Lower bound limit to filter residuals of ramp fit to data read cube.
        sig_clip_res_high: float; default = 5.0
            Upper bound limit to filter residuals of ramp fit to data read cube.
        """

        logging.info(
            "Computing residuals of ramp model from data to estimate variance component of read noise."
        )

        # Initialize ramp residual variance array.
        self.ramp_res_var = np.zeros(
            (self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32
        )
        self.residual_cube = self.data_cube.ramp_model - self.data_cube.data

        self.clipped_res_cube = sigma_clip(
            self.residual_cube,
            sigma_lower=sig_clip_res_low,
            sigma_upper=sig_clip_res_high,
            cenfunc=np.median,
            axis=0,
            masked=True,
            copy=False,
        )
        # Masked = True in sigma clip now returns a masked array with any clipped values being removed
        # from the returned list.

        contains_nans_residual = np.isnan(self.clipped_res_cube).any()
        print("clipped_res_cube contains NaNs:", contains_nans_residual)
        # Count the total number of NaNs in residual_cube
        num_nans_residual = np.isnan(self.clipped_res_cube).sum()
        print("Number of NaNs in clipped_res_cube:", num_nans_residual)

        std = np.std(self.clipped_res_cube, axis=0)
        self.ramp_res_var = np.float32(std)
        return self.ramp_res_var

    def comp_cds_noise(self, sig_clip_cds_low=5.0, sig_clip_cds_high=5.0):
        """
        Compute the correlated double sampling as a noise estimate. The method get_cds_noise() calculates the
        correlated double sampling between pairs of reads in the data cube as a noise term from the standard deviation
        of the differences from all read pairs.

        Intended to be accessible to a user to produce the readnoise reference files

        Parameters
        ----------
        sig_clip_cds_low: float; default = 5.0
            Lower bound limit to filter difference cube.
        sig_clip_cds_high: float; default = 5.0
            Upper bound limit to filter difference cube
        """
        # TODO Optional method accessible to a user to produce the readnoise reference files
        logging.info("Calculating CDS noise.")

        read_diff_cube = np.zeros(
            (
                math.ceil(self.data_cube.num_reads / 2),
                self.data_cube.num_i_pixels,
                self.data_cube.num_j_pixels,
            ),
            dtype=np.float32,
        )
        for i_read in range(0, self.data_cube.num_reads - 1, 2):
            # Avoid index error if num_reads is odd and disregard the last read because it does not form a pair.
            logging.debug(
                f"Calculating correlated double sampling between frames {i_read} and {i_read + 1}"
            )
            rd1 = (
                self.data_cube.ramp_model[i_read, :, :]
                - self.data_cube.data[i_read, :, :]
            )
            rd2 = (
                self.data_cube.ramp_model[i_read + 1, :, :]
                - self.data_cube.data[i_read + 1, :, :]
            )
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1
        clipped_diff_cube = sigma_clip(
            read_diff_cube,
            sigma_lower=sig_clip_cds_low,
            sigma_upper=sig_clip_cds_high,
            cenfunc=np.mean,
            axis=0,
            masked=False,
            copy=False,
        )
        self.cds_noise = np.std(clipped_diff_cube, axis=0)
        return self.cds_noise

    def apply_gain_from_crds(self,
                             tmp_crds_path='./',
                             crds_server="https://roman-crds-test.stsci.edu",
                             crds_context=None):

        """
        # TODO Ru sent the below on 12/9 for follow-up
        from crds.core import config as crds_config
        old_state = crds_config.get_crds_state()
        new_state = dict(old_state)

        # optional..probably not advisable unless you are changing observatories and CRDS paths etc
        # crds_config.clear_crds_state()

        # these key-vals can now be set
        new_state["CRDS_SERVER_URL"] = "https://roman-crds-tvac.stsci.edu"
        crds_config.set_crds_state(new_state)

        The CRDS test suite changes configuration quite a bit during a single pytest run, and we do
        something similar to the above
        https://github.com/spacetelescope/crds/blob/f7cad6ca48bced760351dd52985e9cc5e00b92b0/conftest.py#L146

        # end of Ru message and notes!

        This method downloads all of the gain reference files from CRDS that is context and server specific to
        local directory. The default behavior is to get the most recent files with the closest useafter and the
        roman-crds-test.stsci.edu server. Once the gain files are downloaded into the default scratch directory,
        the detector in the ReadNoise module is matched to the gain file, opened, and the readnoise array is
        multiplied by the gain values per pixel.

        Parameters
        ----------
        tmp_crds_path: string, default = '/';
            Path to download files from CRDS. Default set to cwd or pwd.
        crds_server: string, default = "https://roman-crds-test.stsci.edu"
            Roman CRDS server. Test, Operations, TVAC are allowed.
        crds_context: default = None;
            Specify a context. If context is None, the most recent context is used if None provided.
        """

        # Check if the provided crds_path exists, if not use the current directory
        if not os.path.exists(tmp_crds_path):
            print(f"Directory {tmp_crds_path} does not exist. Using the current directory instead.")
            tmp_crds_path = './'
        os.environ["CRDS_PATH"] = tmp_crds_path
        print('The path to where gain files will be downloaded is: ', os.environ.get("CRDS_PATH"))

        if crds_context is None:
            crds_context = crds.get_default_context()
        print('The CRDS context is: ', crds_context)

        # TODO cant seem to figure out how to set the CRDS server in python to override the environment variable
        # TODO and then get the proper context; switching from TVAC to TEST
        # print(os.environ.get("CRDS_SERVER_URL"))
        # os.environ["CRDS_SERVER_URL"] = crds_server
        # print('The CRDS server is:')
        # print(os.environ.get("CRDS_SERVER_URL"))

        print('Accessing the CRDS server:', os.environ.get("CRDS_SERVER_URL"))

        # Need to get all of the gain files from the default context from CRDS on disk. Need all of the file names and
        # need the context those files are associated with in order to download only the most recent or appropriate
        # gain reference files.
        gain_files_crds = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('gain').reference_names()
        crds_context = crds.get_default_context()
        results = api.dump_references(crds_context, gain_files_crds)
        # results is a dictionary with {filename: 'path to download file' + filename}
        # example: 'roman_wfi_gain_0001.asdf': crds_path + '/references/roman/wfi/roman_wfi_gain_0001.asdf'

        current_time_iso = Time(datetime.datetime.now(), format="datetime").isot
        parameters = {'ROMAN.META.INSTRUMENT.NAME': 'WFI',
                      'ROMAN.META.INSTRUMENT.DETECTOR': self.meta_data.instrument_detector,
                      'ROMAN.META.EXPOSURE.TYPE': self.meta_data.type,
                      'ROMAN.META.EXPOSURE.START_TIME': current_time_iso}

        # Call getrecommendations to get the reference file name for the specific detector and exposure type: i.e. mode
        # example: WFI01 and imaging mode WFI_IMAGE (WIM) or spectral mode WFI_GRISM or WFI_PRISM (WSM)
        recommendation = crds.getrecommendations(
            parameters=parameters,
            reftypes=['GAIN'],
            context=crds_context,
            observatory="roman",
        )
        gain_file_name = recommendation['gain']

        # Now look through downloaded reference files and get proper match.
        # NOTE - CRDS renames files making this query necessary and not assuming the downloaded files are numerically
        # in the same order as WFI01-WFI18
        gain_file_path = results.get(gain_file_name)
        print(gain_file_path)

        # Open the gain file to get the data array and multiply the readnoise_image from the data cube by the gain
        # per pixel
        with asdf.open(gain_file_path) as af:
            gain_meta = af.tree['roman']['meta']
            print(gain_meta)
            gain_image = af.tree['roman']['data']

        if isinstance(gain_image, u.Quantity):  # Only access data from quantity object.
            gain_image = gain_image.value
            logging.debug("Quantity object detected. Extracted data values.")

        self.readnoise_image *= gain_image

    def calculate_error(self):
        """
        Abstract method not applicable to Read Noise.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not applicable to Read Noise.
        """
        pass

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the read noise object from the data model.
        readnoise_datamodel_tree = rds.ReadnoiseRef()
        readnoise_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        readnoise_datamodel_tree["data"] = (
            self.readnoise_image.astype(np.float32) * u.DN
        )

        return readnoise_datamodel_tree

    class ReadNoiseDataCube(DataCube):
        """
        ReadNoiseDataCube class derived from DataCube.
        Handles ReadNoise specific cube calculations
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

