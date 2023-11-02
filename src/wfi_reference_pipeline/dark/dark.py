import roman_datamodels.stnode as rds
import numpy as np
import os
import gc
import asdf
import datetime
import logging
from ..utilities.reference_file import ReferenceFile
from ..utilities.logging_functions import configure_logging
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy import units as u
from wfi_reference_pipeline.constants import WFI_MODE_WIM, WFI_MODE_WSM, WFI_TYPE_IMAGE, WFI_FRAME_TIME
import datetime

configure_logging('dark_dev', path='/grp/roman/RFP/DEV/logs/')


class Dark(ReferenceFile):
    """
    Class Dark() inherits the ReferenceFile() base class methods where static meta data for all reference
    file types are written. Under automated operations conditions, a list of dark calibration files from a directory
    will be the input data for the class to begin generating a dark reference file. A super dark cube of all
    available reads and exposures will go through outlier rejection and then averaging per a multi-accumulation (MA)
    prescription that is retrieved from the RTB database. The resampled MA-table dark cube is analyzed further to
    flag hot and warm pixels. Statistics on these or additional quantities can be written to the RTB database for
    detector performance monitoring with time and comparison across WFI. The dark reference file created is then
    written to disk.
    """

    def __init__(self, dark_file_list, meta_data, bit_mask=None, outfile='roman_dark.asdf', clobber=False,
                 input_dark_cube=None):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------
        dark_file_list: string object; default = None
            List of dark calibration filenames with absolute paths. If no file list is provided, an input dark read cube
            should be supplied.
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer array for supplying a mask for the creation of the dark reference file.
        outfile: string; default = roman_dark.asdf
            Filename with path for saved dark reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        input_dark_cube: numpy array; default = None
            Cube of dark reads to be resampled into MA table specific dark reference file. Dimensions of
            ni x ni x n_reads, where ni is the number of pixels of a square sub-array of the detector by the number of
            reads (n_reads) in the integration. NOTE - For parallelization only square arrays allowed.
        -------
        self.input_data: variable;
            The first positional variable in the Dark class instance assigned in base class ReferenceFile().
            For Dark() self.input_data is a list of string filenames with paths.
        """

        # Access methods of base class ReferenceFile
        super().__init__(dark_file_list, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=True)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI dark reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'DARK'

        logging.info(f'Default dark reference file object: {outfile} ')

        # Initialize attributes
        self.outfile = outfile
        # Additional object attributes
        self.dark_read_cube = input_dark_cube  # Supplied input dark read cube.
        self.super_dark = None  # A cube of all available reads that is sigma clipped and averaged.
        self.read_pattern = None  # read pattern from ma table meta data - nested list of lists reads in resultants
        self.ma_table_sequence = []  # For general treatment of unevenly spaced resultant averaging.
        self.resampled_dark_cube = None  # MA table averaged resultant cube.
        self.resampled_dark_cube_model = None  # MA table resultant ramp model.
        self.resampled_dark_cube_err = None  # MA table averaged resultant error cube.
        self.resultant_tau_arr = None  # Variance-based resultant time tau_i from Casterano et al. 2022 equation 14.
        self.dark_rate_image = None  # Rate image from ramp fit.
        self.dark_intercept_image = None  # Intercept image from ramp fit.
        self.dark_rate_var = None  # Variance in fitted dark rate image.
        self.dark_intercept_var = None  # Variance in fitted dark intercept image.
        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_reads = None  # Number of reads in data cube being analyzed.
        self.ni = None  # Number of pixels.
        self.frame_time = None  # Frame time from ancillary data.
        self.time_arr = None  # Time array of an exposure.

    def make_super_dark(self, sig_clip_md_low=3.0, sig_clip_md_high=3.0):
        """
        The method super() ingests all files located in a directory as a python object list of
        filenames with absolute paths. A super dark is created by iterating through each read of every
        dark calibration file, read by read (see NOTE below). A cube of reads is formed into a numpy array and sigma
        clipped and the mean of the clipped data cube is saved as the super dark class attribute.

        NOTE: The algorithm is file I/O intensive but utilizes less memory while performance is only marginally slower.
        Initial testing was performed by R. Cosentino with 12 dark files that each had 21 reads where this method
        took ~330 seconds and had a peak memory usage of 2.5 GB. Opening and loading all files at once took ~300
        seconds with a peak memory usage of 36 GB. Running from the command line or in the ipython intrepreter
        displays significant differences in memory usage and run time.

        Parameters
        ----------
        sig_clip_md_low: float; default = 3.0
            Lower bound limit to filter data.
        sig_clip_md_high: float; default = 3.0
            Upper bound limit to filter data
        """

        # Display the directory name where the dark calibration files are located to make the super dark.
        logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to construct super dark object.')

        # Find the dark calibration file with the most number of reads to initialize the super dark cube.
        tmp_reads = []
        for fl in range(0, len(self.input_data)):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
            tmp_reads.append(n_rds)
            tmp.close()
        num_reads_set = [*set(tmp_reads)]
        del tmp_reads, tmp
        gc.collect()

        # The super dark length is the maximum number of reads in all dark calibration files to be used
        # when creating the dark reference file. Need to try over files with different lengths
        # to compute average read by read for all files
        self.super_dark = np.zeros((np.max(num_reads_set), 4096, 4096), dtype=np.float32)
        # This method of opening and closing each file read by read is file I/O intensive however
        # it is efficient on memory usage.
        logging.info(f'Reading dark asdf files read by read to compute average for super dark.')
        print('reading files')
        for rd in range(0, np.max(num_reads_set)):
            dark_read_cube = []
            logging.info(f'On read {rd} of {np.max(num_reads_set)}')
            print('read', rd)
            for fl in range(0, len(self.input_data)):
                print(fl, 'file')
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                rd_tmp = tmp.tree['roman']['data']
                dark_read_cube.append(rd_tmp[rd, :, :])
                del tmp, rd_tmp
                gc.collect()  # clean up memory
            clipped_reads = sigma_clip(dark_read_cube, sigma_lower=sig_clip_md_low, sigma_upper=sig_clip_md_high,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
            self.super_dark[rd, :, :] = np.mean(clipped_reads, axis=0)
            del clipped_reads
            gc.collect()  # Clean up memory.

        # set reference pixel border to zero for super dark
        # this needs to be done differently for multi sub array jigsaw handling
        # move to when making the mask and final stitching together different pieces to do the border
        self.super_dark[:, :4, :] = 0.
        self.super_dark[:, -4:, :] = 0.
        self.super_dark[:, :, :4] = 0.
        self.super_dark[:, :, -4:] = 0.
        logging.info(f'Super dark attribute created.')

    def save_suoer_dark(self, md_outfile=None):
        """
        The method save_super_dark with default conditions will write the super dark cube into an asdf
        file for each detector in the directory from which the input files where pointed to and used to
        construct the super dark read cube. A user can specify an absolute path or relative file string
        to write the super dark file name to disk.

        Parameters
        ----------
        md_outfile: str; default = None
            File string. Absolute or relative path for optional input.
            By default, None is provided but the method below generates the asdf file string from meta
            data of the input files such as date and detector number  (i.e. WFI01) in the filename.
        """

        meta_md = {'pedigree': "GROUND", 'description': "Super dark internal reference file calibration product"
                                                        "generated from Reference File Pipeline.",
                   'date': Time(datetime.datetime.now()), 'detector': self.meta['instrument']['detector']}
        if md_outfile is None:
            md_outfile = os.path.dirname(self.input_data[0]) + '/' + meta_md['detector'] + '_super_dark.asdf'
        else:
            md_outfile = 'roman_super_dark.asdf'
        self.check_output_file(md_outfile)
        logging.info(f'Saving super dark to disk.')

        af = asdf.AsdfFile()
        af.tree = {'meta': meta_md, 'data': self.super_dark}
        af.write_to(md_outfile)

    def initialize_arrays(self, num_resultants=None, ni=None):
        """
        Method initialize_arrays makes arrays of the dimensions of the dark_read_cube, which are also required
        in the data model.

        Parameters
        ----------
        num_resultants: integer; Default=None
            The number of resultants
        ni: integer; Default=None
            Number of square pixels of array ni. Cubes are num_resultants x ni x ni.
        """

        # Flow control and logging messaging depending on how the Dark() class is instantiated
        if self.input_data is not None and self.dark_read_cube is None:
            self.dark_read_cube = self.super_dark
            logging.info(f'Super dark created from input file list used for MA table resampling.')
        elif self.dark_read_cube is not None:
            logging.info(f'Input dark read cube being used.')
        else:
            raise ValueError('No data supplied to make dark reference file for MA table resampling!')

        #TODO discuss parallelizatino strategy
        self.n_reads, self.ni, _ = np.shape(self.dark_read_cube)
        self.resampled_dark_cube = np.zeros((num_resultants, self.ni, self.ni), dtype=np.float32)
        self.dark_rate_image = np.zeros((self.ni, self.ni), dtype=np.float32)
        self.dark_rate_var = np.zeros((self.ni, self.ni), dtype=np.float32)
        logging.info(f'Error arrays with number of resultants initialized with zeros.')

        # Make the time array for the length of the dark read cube exposure.
        if self.meta['exposure']['type'] == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WSM]  # frame time in spectral mode in seconds
        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.n_reads} reads long with a frame '
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.n_reads + 1)])
        self.resultant_tau_arr = np.zeros(num_resultants, dtype=np.float32)

    def make_ma_table_resampled_dark(self, read_pattern=None, num_resultants=None, num_rds_per_res=None):
        """
        The method make_ma_table_resampled_dark() uses the input read_pattern, which is a nested list of lists,
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
            num_resultants = len(read_pattern)
            self.initialize_arrays(num_resultants)
            # Iterate over each nested list in the read pattern
            for res_i, read_pattern_frames in enumerate(read_pattern):
                # Get the average time for the list of frames in the read pattern
                read_pattern_zero_indices = [i - 1 for i in read_pattern_frames]  # zero index for time array
                self.resultant_tau_arr[res_i] = np.mean(self.time_arr[read_pattern_zero_indices])
                # Average the data by summing read by read and dividing by number of raeds
                for read_i in read_pattern_frames:
                    self.resampled_dark_cube[res_i] += self.dark_read_cube[read_i - 1]  # Adjusted for 0 indexing
                self.resampled_dark_cube[res_i] /= len(read_pattern_frames)
        else:
            # Use even spacing resultant and reads per resultant provided to the method and
            # get mean time of resultant for tau array
            if not isinstance(num_resultants, int) or not isinstance(num_rds_per_res, int):
                raise ValueError("Both num_resultants and num_rds_per_res must be integers.")
            if num_resultants is None or num_rds_per_res is None:
                raise ValueError("Both num_resultants and num_rds_per_res must be provided simultaneously.")
            print("Averaging with even spacing.")
            self.initialize_arrays(num_resultants)
            if num_rds_per_res > self.n_reads:
                raise ValueError('Cannot average over more reads than supplied in the dark read cube.')
            # Averaging over reads per ma table specs or user defined even spacing.
            logging.info(f'Averaging over reads with evenly spaced resultants.')
            for res_i in range(num_resultants):
                i1 = res_i * num_rds_per_res
                i2 = i1 + num_rds_per_res
                if i2 > self.n_reads:
                    logging.info(f'Warning: The number of reads per resultant was not evenly divisible into the number'
                                 f' of available reads to average and remainder reads were skipped.')
                    logging.info(f'Resultants after resultant {res_i+1} contain zeros.')
                    break  # Remaining reads cannot be evenly divided
                self.resampled_dark_cube[res_i, :, :] = np.mean(self.dark_read_cube[i1:i2, :, :], axis=0)
                self.resultant_tau_arr[res_i] = np.mean(self.time_arr[i1:i2])
            logging.info(f'MA table resampling with {num_resultants} resultants averaging {num_rds_per_res}'
                         f' reads per resultant complete.')

    def fit_dark_ramp(self):
        """
        The fit_dark_ramp() method computes the fitted ramp or slope along the time axis for the resultants in the
        resampled_dark_cube attribute using a 1st order polyfit. The best fit solutions and variance are saved into
        attributes.
        """

        logging.info(f'Computing dark rate image.')
        # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.

        p, c = np.polyfit(self.resultant_tau_arr,
                          self.resampled_dark_cube.reshape(len(self.resultant_tau_arr), -1), 1, full=False, cov=True)

        # Reshape results back to 2D arrays.
        self.dark_rate_image = p[0].reshape(self.ni, self.ni).astype(np.float32)  # the fitted ramp slope image
        self.dark_rate_var = c[0, 0, :].reshape(self.ni, self.ni).astype(np.float32)  # covariance matrix slope variance
        # If needed the dark intercept image and variance are p[1] and c[1,1,:]

    def calculate_dark_error(self):
        """
        #TODO re-evaluate the error propagation for dark and how fit_dark_ramp() and this method operate and computational performance
        Old method of computing dark error
        """

        # Generate a dark ramp cube model per the resampled ma table specs.
        self.resampled_dark_cube_model = np.zeros((len(self.resampled_dark_cube), self.ni, self.ni), dtype=np.float32)
        for tt in range(0, len(self.resultant_tau_arr)):
            self.resampled_dark_cube_model[tt, :, :] = self.dark_rate_image * self.resultant_tau_arr[tt] \
                                                       + self.dark_intercept_image  # y = m*x + b
        # Calculate the residuals of the dark ramp model and the data
        residual_cube = self.resampled_dark_cube_model - self.resampled_dark_cube
        std = np.std(residual_cube, axis=0)  # this is the standard deviation of residuals from the resampled dark cube
        # model and the resampled dark cube data, std^2 is therefore the resampled read noise variance
        # the dark cube error array should be a 2D image of 4096x4096 with the slope variance from the model fit
        # and the variance of the resampled residuals are added in quadrature
        self.resampled_dark_cube_err[0, :, :] = (std * std + self.dark_rate_var) ** 0.5

    def update_dq_mask(self, dead_pixel_rate=0.0001, hot_pixel_rate=0.015, warm_pixel_rate=0.010):
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

        logging.info('Flagging hot and warm pixels and updating DQ array.')
        # Locate hot and warm pixel ni,nj positions in 2D array
        self.dead_pixels = np.where(self.dark_rate_image < dead_pixel_rate)
        self.hot_pixels = np.where(self.dark_rate_image >= hot_pixel_rate)
        self.warm_pixels = np.where((warm_pixel_rate <= self.dark_rate_image) & (self.dark_rate_image < hot_pixel_rate))

        # Set mask DQ flag values
        self.mask[self.dead_pixels] += self.dqflag_defs['DEAD']
        self.mask[self.hot_pixels] += self.dqflag_defs['HOT']
        self.mask[self.warm_pixels] += self.dqflag_defs['WARM']

    def make_metrics_dicts(self):
        """
        The method make_metrics_dicts is used to create reference file type specific
        metrics for tracking by the data monitoring tool from entries into the
        the RTB Database.
        """

        # Create the dark file dictionary
        db_dark_fl_dict = {'detector': self.meta['instrument']['detector'],
                            'exposure_type': self.meta['exposure']['type'],
                            'created_date': datetime.datetime.utcnow(),
                            'use_after': datetime.datetime.utcnow(),
                            'crds_filename': 'test_crds_filename.asdf',
                            'crds_delivery_id': 1}

        # Get the number of dead, hot, and warm pixels for metric tracking
        _, num_dead_pixels = np.shape(self.hot_pixels)
        _, num_hot_pixels = np.shape(self.hot_pixels)
        _, num_warm_pixels = np.shape(self.warm_pixels)
        logging.info(f'Found {num_hot_pixels} hot pixels and {num_warm_pixels} warm pixels in dark rate ramp image.')

        # Create the dark dq dictionary
        db_dark_dq_dict = {'num_dead_pix': num_dead_pixels,
                        'num_hot_pix': num_hot_pixels,
                        'num_warm_pix': num_warm_pixels,
                        'hot_pix_rate': 0.015,
                        'warm_pix_rate': 0.010}

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
            'amp_id': list(range(1, num_amps + 1)),
            'median_dark_current': median_values,
            'mean_dark_current': mean_values
        }

        # Create the dark structure dictionary
        db_dark_struc_dict = {'coefficient': 42.0}

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
        dark_datamodel_tree['meta'] = self.meta
        dark_datamodel_tree['data'] = self.resampled_dark_cube * u.DN
        dark_datamodel_tree['dark_slope'] = self.dark_rate_image * u.DN / u.s
        dark_datamodel_tree['dark_slope_error'] = (self.dark_rate_var ** 0.5) * u.DN / u.s
        dark_datamodel_tree['dq'] = self.mask

        return dark_datamodel_tree

    def save_dark(self, datamodel_tree=None):
        """
        The method save_dark writes the reference file object to the specified asdf outfile.
        """

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
