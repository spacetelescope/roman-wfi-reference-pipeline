import logging
import datetime
import numpy as np
import roman_datamodels.stnode as rds
import roman_datamodels as rdm
from astropy import units as u

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

    Sample Dark Run From Superdark:
    dark = Dark(meta_data, file_list=superdark.asdf, ...)
    dark.make_ma_table_resampled_data()
    dark.calculate_error() # TODO - make abstract method
    dark.update_dq_mask()  # TODO - make abstract method
    -> QC call here
    dark.generate_outfile()

    Sample Dark Run From 3D Cube:
    dark = Dark(meta_data, ref_type_data=user_cube, ...)
    dark.make_ma_table_resampled_data(None, None, user_readpattern) OR dark.make_ma_table_resampled_data(None, num_resultants, num_reads_per_resultant)
    dark.calculate_error() # TODO - make abstract method
    dark.update_dq_mask()  # TODO - make abstract method
    dark.generate_outfile()

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

        logging.debug(f"Default dark reference file object: {outfile} ")

        # Module flow creating reference file
        # This SHOULD only be one file in the file list, and it is the SuperDark file
        if self.file_list:
            # Get file list properties and select data cube.
            if len(self.file_list) > 1:
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
                logging.info('Must call make_ma_table_resampled_cube and then make_dark_rate_image() to '
                             'finish creating reference file.')
            else:
                raise ValueError('Input data is not a valid numpy array of dimension 3.')

        # MA Table attributes
        # TODO populate from database or MA Table Config file?
        self.ma_table_read_pattern = 0  # read pattern from ma table meta data - nested list of lists reads in resultants will be replacing (ngroups, nframes, groupgap)
        self.num_resultants = 0 # length of self.ma_table_read_pattern
        self.resampled_data = None
        self.resampled_data_err = None
        self.resampled_model = None
        self.resultant_tau_arr = None  # Variance-based resultant time tau_i from Casterano et al. 2022 equation 14
        self.hot_pixel_rate = 0
        self.warm_pixel_rate = 0
        self.dead_pixel_rate = 0


        # self.resampled_data = np.zeros((self.num_resultants, self.num_i_pixels, self.num_j_pixels), dtype=np.float32)
        # self.resampled_data_err = np.zeros((self.num_resultants, self.num_i_pixels, self.num_j_pixels), dtype=np.float32)
        # self.intercept_image = np.zeros((self.num_i_pixels, self.num_j_pixels), dtype=np.float32) # Intercept image from ramp fit.
        # self.intercept_var = np.zeros((self.num_i_pixels, self.num_j_pixels), dtype=np.float32) # Variance in fitted intercept image.
        # self.resultant_tau_arr = np.zeros(self.num_resultants, dtype=np.float32)



    def _get_superdark_from_file_list(self):
        """
        Method to open superdark asdf file from file_list
        inside of file list send into Dark()

        """

        logging.info("OPENING - " + self.file_list) # file_list is already checked to be single value
        data = rdm.open(self.file_list)
        if isinstance(data, u.Quantity):  # Only access data from quantity object.
            data = data.value
        return DarkDataCube(data, self.meta_data.type)


    def make_ma_table_resampled_data(self,
                                     num_resultants=None,
                                     num_reads_per_resultant=None,
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
        num_reads_per_resultant: integer; Default=None
            The user supplied number of reads per resultant in evenly spaced resultants.
        """

        if read_pattern:
            # Use read pattern for resampling by averaging reads into resultants and
            # get mean time of resultant for tau array
            self.num_resultants = len(read_pattern)
            # Iterate over each nested list in the read pattern
            logging.debug('Averaging over reads following read pattern supplied.')
            for resultant_i, read_pattern_frames in enumerate(read_pattern):
                # Get the average time for the list of frames in the read pattern
                read_pattern_zero_indices = [i - 1 for i in read_pattern_frames]  # zero index for time array
                self.resultant_tau_arr[resultant_i] = np.mean(self.time_arr[read_pattern_zero_indices]) # TODO - do we need this? DMS calculates this for us
                # Average the data by summing read by read and dividing by number of raeds
                for read_i in read_pattern_frames:
                    self.resampled_data[resultant_i] += self.data_cube.data[read_i - 1]  # Adjusted for 0 indexing
                self.resampled_data[resultant_i] /= len(read_pattern_frames)
            logging.debug('Finished re-sampling with read pattern.')
        else:
            # Use even spacing resultant and reads per resultant provided to the method and
            # get mean time of resultant for tau array
            if not isinstance(num_resultants, int) or not isinstance(num_reads_per_resultant, int):
                raise ValueError("Both num_resultants and num_rds_per_res must be integers.")
            if num_resultants is None or num_reads_per_resultant is None:
                raise ValueError("Both num_resultants and num_rds_per_res are required inputs.")
            logging.debug('Averaging over reads with evenly spaced resultants.')
            self.num_resultants = num_resultants
            if num_reads_per_resultant > self.data_cube.num_reads:
                raise ValueError('Cannot average over more reads than supplied in the dark cube.')
            # Averaging over reads per ma table specs or user defined even spacing.
            for resultant_i in range(self.num_resultants):
                i1 = resultant_i * num_reads_per_resultant
                i2 = i1 + num_reads_per_resultant
                if i2 > self.data_cube.num_reads:
                    logging.warning('Warning: The number of reads per resultant was not evenly divisible into the number'
                                 ' of available reads to average and remainder reads were skipped.')
                    logging.warning(f'Resultants after resultant {resultant_i+1} contain zeros.')
                    break  # Remaining reads cannot be evenly divided
                self.resampled_data[resultant_i, :, :] = np.mean(self.data_cube.data[i1:i2, :, :], axis=0)
                self.resultant_tau_arr[resultant_i] = np.mean(self.data_cube.time_arr[i1:i2])

            logging.info(f'MA table resampling with {self.num_resultants} resultants averaging {num_reads_per_resultant}'
                         f' reads per resultant complete.')


    def calculate_dark_error(self):
        """
        #TODO re-evaluate the error propagation for dark and how fit_dark_ramp() and this method operate and computational performance
        Old method of computing dark error
        """

        # Generate a dark ramp cube model per the resampled ma table specs.
        self.resampled_model = np.zeros((len(self.resampled_data), self.data_cube.num_i_pixels, self.data_cube.num_j_pixels), dtype=np.float32
        )
        for tt in range(0, len(self.resultant_tau_arr)):
            self.resampled_model[tt, :, :] = (
                self.data_cube.rate_image * self.resultant_tau_arr[tt]
                + self.data_cube.intercept_image
            )  # y = m*x + b
        # Calculate the residuals of the dark ramp model and the data
        residual_cube = self.resampled_model - self.resampled_data
        std = np.std(
            residual_cube, axis=0
        )
        # This is the standard deviation of residuals from the resampled cube
        # model and the resampled cube data. Therefore std^2 is the resampled read noise variance.
        # The dark cube error array should be a 2D image of 4096x4096 with the slope variance from the model fit
        # and the variance of the resampled residuals are added in quadrature.
        self.resampled_data_err[0, :, :] = (std * std + self.data_cube.rate_image_err) ** 0.5

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
        self.mask[self.data_cube.rate_image > self.hot_pixel_rate] += self.dqflag_defs['HOT']
        self.mask[(self.warm_pixel_rate <= self.data_cube.rate_image) & (self.data_cube.rate_image < self.hot_pixel_rate)] \
            += self.dqflag_defs['WARM']
        self.mask[self.data_cube.rate_image < self.dead_pixel_rate] += self.dqflag_defs['DEAD']

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
        warm_pixel_mask = (self.warm_pixel_rate <= self.data_cube.rate_image) & (self.data_cube.rate_image < self.hot_pixel_rate)
        num_warm_pixels = np.sum(warm_pixel_mask)
        dead_pixel_mask = self.data_cube.rate_image < self.dead_pixel_rate
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

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        dark_datamodel_tree = rds.DarkRef()
        dark_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        dark_datamodel_tree['data'] = self.resampled_data * u.DN
        dark_datamodel_tree['dark_slope'] = self.data_cube.rate_image * u.DN / u.s
        dark_datamodel_tree['dark_slope_error'] = (self.data_cube.rate_image_err ** 0.5) * u.DN / u.s
        dark_datamodel_tree['dq'] = self.mask

        return dark_datamodel_tree
