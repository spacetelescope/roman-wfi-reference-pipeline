import logging
import datetime
import numpy as np
import roman_datamodels.stnode as rds
import roman_datamodels as rdm
from astropy import units as u
from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_MODE_WSM,
    WFI_TYPE_IMAGE,
)
from wfi_reference_pipeline.reference_types.data_cube import DarkDataCube
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
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_dark.asdf",
        clobber=False
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
        data_array: numpy array; default = None
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

        # Default meta creation for moedule specific ref type.
        if not isinstance(meta_data, WFIMetaDark):
            raise TypeError(f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDark")
        if len(self.meta_data.description) == 0:
            self.meta_data.description = 'Roman WFI dark reference file.'

        logging.info(f"Default dark reference file object: {outfile} ")

        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.n_files = len(self.file_list)
            if self.n_files > 1:
                raise ValueError('A single super dark was expected in file_list..')
            else:
                self.data_cube = self._get_superdark_from_file_list()
            # Must make_ma_table_resampled_cube and then make_dark_rate_image()
        else:
            if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
                raise TypeError("Input data is neither a numpy array nor a Quantity object.")
            if isinstance(ref_type_data, u.Quantity):  # Only access data from quantity object.
                ref_type_data = ref_type_data.value
                logging.info('Quantity object detected. Extracted data values.')
            dim = ref_type_data.shape
            if len(dim) == 3:
                logging.info('User supplied 3D data cube to make dark reference file.')
                self.data_cube = DarkDataCube(ref_type_data, self.meta_data.type)
                # Must make_ma_table_resampled_cube and then make_dark_rate_image()
                logging.info('Must call  make_ma_table_resampled_cube and then make_dark_rate_image() to '
                             'finish creating reference file.')
            else:
                raise ValueError('Input data is not a valid numpy array of dimension 3.')

        self.superdark = None
        # Attributes to make reference file with valid data model.
        self.resampled_data_cube = None  # The attribute 'data' in data model.
        self.dark_rate_image = None  # The attribute 'dark_slope' in data model.
        self.dark_rate_var = None  # The attribute 'dark_slope_error' in data model.
        # Others populated but not incorporated into data model.
        self.resampled_dark_cube_err = None  # MA table averaged resultant error cube.
        self.dark_intercept_image = None  # Intercept image from ramp fit.
        self.dark_intercept_var = None  # Variance in fitted intercept image.

        # MA Table attributes
        self.read_pattern = None  # read pattern from ma table meta data - nested list of lists reads in resultants
        self.ma_table_sequence = []  # For general treatment of unevenly spaced resultant averaging.
        self.resultant_tau_arr = None  # Variance-based resultant time tau_i from Casterano et al. 2022 equation 14
        self.num_resultants = None
        self.resampled_model = None



    def _get_superdark_from_file_list(self):
        """
        Method to open superdark asdf file from first positional location
        inside of file list send into Dark()

        return: self.superdark
        """

        logging.info("OPENING - " + self.file_list[0])
        superdark = rdm.open(self.file_list[0])
        if isinstance(superdark, u.Quantity):  # Only access data from quantity object.
            self.superdark = superdark.value
        return self.superdark

    def _initialize_arrays(self):
        """
        Method to initialize arrays that are written to the data model attributes and
        could be used in the Cube class method accessible to all reference file types.

        Parameters
        ----------
        num_resultants: integer; Default=None
            The number of resultants
        """


        #TODO discuss parallelizatino strategy and cube class
        self.num_reads, self.data_cube.num_i_pixels, self.data_cube.num_j_pixels = np.shape(self.data_cube)

        #TODO not to be in cube class
        self.resampled_data_cube = np.zeros((self.num_resultants, self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32)
        self.dark_rate_image = np.zeros((self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32)
        self.dark_rate_var = np.zeros((self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32)

        self.resampled_data_cube_err = np.zeros((self.num_resultants, self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32)
        self.dark_intercept_image = np.zeros((self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32)
        self.dark_intercept_var = np.zeros((self.data_cube.num_i_pixels, self.nnum_j_pixelsi), dtype=np.float32)

        self.resultant_tau_arr = np.zeros(self.num_resultants, dtype=np.float32)
        #TODO all of this cube class below
        logging.info('Error arrays with number of resultants initialized with zeros.')
        # Make the time array for the length of the dark read cube exposure.
        if self.meta_data.type == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WSM]  # frame time in spectral mode in seconds
        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.num_reads} reads long with a frame '
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.num_reads + 1)])

    def make_ma_table_resampled_cube(self,
                                     num_resultants=None,
                                     num_rds_per_res=None,
                                     read_pattern=None
                                     ):
        """
        The method make_ma_table_resampled_cube() uses the input read_pattern, which is a nested list of lists,
        or the number of resultants and reads per resultant to average reads into resultants. If read_pattern
        is supplied, the even spacing parameters will be ignored.

        Parameters
        ----------
        read_pattern: list of lists; Default=None
            Nested list of lists with integers for averaging reads into resultants.
        num_resultants: integer; Default=None
            The number of resultants.
        num_rds_per_res: integer; Default=None
            The user supplied number of reads per resultant in evenly spaced resultants.
        """

        if read_pattern:
            # Use read pattern for resampling by averaging reads into resultants and
            # get mean time of resultant for tau array
            self.num_resultants = len(read_pattern)
            self._initialize_arrays()
            # Iterate over each nested list in the read pattern
            logging.info('Averaging over reads following read pattern supplied.')
            for res_i, read_pattern_frames in enumerate(read_pattern):
                # Get the average time for the list of frames in the read pattern
                read_pattern_zero_indices = [i - 1 for i in read_pattern_frames]  # zero index for time array
                self.resultant_tau_arr[res_i] = np.mean(self.time_arr[read_pattern_zero_indices])
                # Average the data by summing read by read and dividing by number of raeds
                for read_i in read_pattern_frames:
                    self.resampled_data_cube[res_i] += self.data_cube[read_i - 1]  # Adjusted for 0 indexing
                self.resampled_data_cube[res_i] /= len(read_pattern_frames)
            logging.info('Finished re-sampling with read pattern.')
        else:
            # Use even spacing resultant and reads per resultant provided to the method and
            # get mean time of resultant for tau array
            if not isinstance(num_resultants, int) or not isinstance(num_rds_per_res, int):
                raise ValueError("Both num_resultants and num_rds_per_res must be integers.")
            if num_resultants is None or num_rds_per_res is None:
                raise ValueError("Both num_resultants and num_rds_per_res are required inputs.")
            logging.info('Averaging over reads with evenly spaced resultants.')
            self.num_resultants = num_resultants
            self._initialize_arrays()
            if num_rds_per_res > self.num_reads:
                raise ValueError('Cannot average over more reads than supplied in the dark cube.')
            # Averaging over reads per ma table specs or user defined even spacing.
            for res_i in range(self.num_resultants):
                i1 = res_i * num_rds_per_res
                i2 = i1 + num_rds_per_res
                if i2 > self.num_reads:
                    logging.info('Warning: The number of reads per resultant was not evenly divisible into the number'
                                 ' of available reads to average and remainder reads were skipped.')
                    logging.info(f'Resultants after resultant {res_i+1} contain zeros.')
                    break  # Remaining reads cannot be evenly divided
                self.resampled_data_cube[res_i, :, :] = np.mean(self.data_cube[i1:i2, :, :], axis=0)
                self.resultant_tau_arr[res_i] = np.mean(self.time_arr[i1:i2])

            logging.info(f'MA table resampling with {self.num_resultants} resultants averaging {num_rds_per_res}'
                         f' reads per resultant complete.')

    def make_dark_rate_image(self):
        """
        This method is used to generate the reference file image type from the file list or a data cube.
        """

        logging.info('Computing dark rate image.')
        # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.

        #TODO move to cube class ?
        p, c = np.polyfit(self.resultant_tau_arr,
                          self.resampled_data_cube.reshape(len(self.resultant_tau_arr), -1), 1, full=False, cov=True)

        # Reshape results back to 2D arrays.
        self.dark_rate_image = p[0].reshape(self.data_cube.num_i_pixels, self.data_cube.num_j_pixels).astype(np.float32)  # the fitted ramp slope image
        self.dark_rate_var = c[0, 0, :].reshape(self.data_cube.num_i_pixels, self.data_cube.num_j_pixels).astype(np.float32)  # covariance matrix slope variance
        # If needed the dark intercept image and variance are p[1] and c[1,1,:]

    def calculate_dark_error(self):
        """
        #TODO re-evaluate the error propagation for dark and how fit_dark_ramp() and this method operate and computational performance
        Old method of computing dark error
        """

        # Generate a dark ramp cube model per the resampled ma table specs.
        self.resampled_model = np.zeros((len(self.resampled_data_cube), self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32
        )
        for tt in range(0, len(self.resultant_tau_arr)):
            self.resampled_model[tt, :, :] = (
                self.dark_rate_image * self.resultant_tau_arr[tt]
                + self.dark_intercept_image
            )  # y = m*x + b
        # Calculate the residuals of the dark ramp model and the data
        residual_cube = self.resampled_model - self.resampled_data_cube
        std = np.std(
            residual_cube, axis=0
        )
        # This is the standard deviation of residuals from the resampled cube
        # model and the resampled cube data. Therefore std^2 is the resampled read noise variance.
        # The dark cube error array should be a 2D image of 4096x4096 with the slope variance from the model fit
        # and the variance of the resampled residuals are added in quadrature.
        self.resampled_dark_cube_err[0, :, :] = (std * std + self.dark_rate_var) ** 0.5

    def update_dq_mask(self, hot_pixel_rate=0.015, warm_pixel_rate=0.010, dead_pixel_rate=0.0001):
        #TODO evaluate options for variabiles like this and sigma clipping with a parameter file?
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

        logging.info('Flagging dead, hot, and warm pixels and updating DQ array.')
        # Locate hot and warm pixel num_i_pixels, num_j_pixels positions in 2D array
        self.mask[self.dark_rate_image > self.hot_pixel_rate] += self.dqflag_defs['HOT']
        self.mask[(self.warm_pixel_rate <= self.dark_rate_image) & (self.dark_rate_image < self.hot_pixel_rate)] \
            += self.dqflag_defs['WARM']
        self.mask[self.dark_rate_image < self.dead_pixel_rate] += self.dqflag_defs['DEAD']

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

        hot_pixel_mask = self.dark_rate_image > self.hot_pixel_rate
        num_hot_pixels = np.sum(hot_pixel_mask)
        warm_pixel_mask = (self.warm_pixel_rate <= self.dark_rate_image) & (self.dark_rate_image < self.hot_pixel_rate)
        num_warm_pixels = np.sum(warm_pixel_mask)
        dead_pixel_mask = self.dark_rate_image < self.dead_pixel_rate
        num_dead_pixels = np.sum(dead_pixel_mask)

        logging.info(f'Found {num_hot_pixels} hot pixels,  {num_warm_pixels} warm pixels, and {num_dead_pixels} were'
                     f'found in the dark rate ramp image.')
        print('hot, warm, dead pixels', num_hot_pixels, num_warm_pixels, num_dead_pixels)

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
            amp_i = self.dark_rate_image[:, start_index:end_index]
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

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        dark_datamodel_tree = rds.DarkRef()
        dark_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        dark_datamodel_tree['data'] = self.resampled_data_cube * u.DN
        dark_datamodel_tree['dark_slope'] = self.dark_rate_image * u.DN / u.s
        dark_datamodel_tree['dark_slope_error'] = (self.dark_rate_var ** 0.5) * u.DN / u.s
        dark_datamodel_tree['dq'] = self.mask

        return dark_datamodel_tree
