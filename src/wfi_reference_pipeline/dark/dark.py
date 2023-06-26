import roman_datamodels.stnode as rds
from roman_datamodels import units as ru
import numpy as np
import os, gc, asdf, datetime, logging
from ..utilities.reference_file import ReferenceFile
from ..utilities.logging_functions import configure_logging
from astropy.stats import sigma_clip
from astropy.time import Time


configure_logging('dark_dev', path='/grp/roman/RFP/DEV/logs/')


class Dark(ReferenceFile):
    """
    Class Dark() inherits the ReferenceFile() base class methods where static meta data for all reference
    file types are written. Under automated operations conditions, a list of dark calibration files from a directory
    will be the input data for the class to begin generating a dark reference file. A master dark cube of all
    available reads and exposures will go through outlier rejection and then averaging per a multi-accumulation (MA)
    prescription that is retrieved from the RTB database. The resampled MA-table dark cube is analyzed further to
    flag hot and warm pixels. Statistics on these or additional quantities can be written to the RTB database for
    detector performance monitoring with time and comparison across WFI. The dark reference file created is then
    written to disk.
    """

    def __init__(self, dark_filelist, meta_data, bit_mask=None, outfile='roman_dark.asdf', clobber=False,
                 input_dark_cube=None):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------
        dark_filelist: string object; default = None
            List of dark calibration filenames with absolute paths. If no file list is provided, an input dark read cube
            should be supplied.
        meta_data: dictionary; default = None
            Dictionary of information for read noise reference file as required by romandatamodels.
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
        super().__init__(dark_filelist, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=True)

        logging.info(f'Default dark reference file object: {outfile} ')

        # Additional object attributes
        self.dark_read_cube = input_dark_cube  # Supplied input dark read cube.
        self.master_dark = None  # A cube of all available reads that is sigma clipped and averaged.
        self.read_pattern = None  # read pattern from ma table meta data - nested list of lists reads in resultants
        self.ma_table_sequence = []  # For general treatment of unevenly spaced resultant averaging.
        self.resampled_dark_cube = None  # MA table averaged resultant cube.
        self.resampled_dark_cube_model = None  # MA table resultant ramp model.
        self.resampled_dark_cube_err = None  # MA table averaged resultant error cube.
        self.resultant_tau_arr = None  # Variance-based resultant time tau_i from Casterano et al. 2022 equation 14.
        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_reads = None  # Number of reads in data cube being analyzed.
        self.ni = None  # Number of pixels.
        self.frame_time = None  # Frame time from ancillary data.
        self.time_arr = None  # Time array of an exposure.

        if self.input_data is None and self.dark_read_cube is None:
            raise ValueError('No data supplied to make dark reference file!')

    def make_master_dark(self, sig_clip_md_low=3.0, sig_clip_md_high=3.0):
        """
        The method make_master_dark() ingests all files located in a directory as a python object list of
        filenames with absolute paths. A master dark is created by iterating through each read of every
        dark calibration file, read by read (see NOTE below). A cube of reads is formed into a numpy array and sigma
        clipped and the mean of the clipped data cube is saved as the master dark class attribute.

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

        # Display the directory name where the dark calibration files are located to make the master dark.
        logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to construct master dark object.')

        # Find the dark calibration file with the most number of reads to initialize the master dark cube.
        tmp_reads = []
        for fl in range(0, len(self.input_data)):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
            tmp_reads.append(n_rds)
            tmp.close()
        num_reads_set = [*set(tmp_reads)]
        del tmp_reads, tmp
        gc.collect()

        # The master dark length is the maximum number of reads in all dark calibration files to be used
        # when creating the dark reference file. Need to try over files with different lengths
        # to compute average read by read for all files
        self.master_dark = np.zeros((np.max(num_reads_set), 4096, 4096), dtype=np.float32)
        # This method of opening and closing each file read by read is file I/O intensive however
        # it is efficient on memory usage.
        logging.info(f'Reading dark asdf files read by read to compute average for master dark.')
        for rd in range(0, np.max(num_reads_set)):
            dark_read_cube = []
            logging.info(f'On read {rd} of {np.max(num_reads_set)}')
            for fl in range(0, len(self.input_data)):
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                rd_tmp = tmp.tree['roman']['data']
                dark_read_cube.append(rd_tmp[rd, :, :])
                del tmp, rd_tmp
                gc.collect()  # clean up memory
            clipped_reads = sigma_clip(dark_read_cube, sigma_lower=sig_clip_md_low, sigma_upper=sig_clip_md_high,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
            self.master_dark[rd, :, :] = np.mean(clipped_reads, axis=0)
            del clipped_reads
            gc.collect()  # Clean up memory.

        # set reference pixel border to zero for master dark
        # this needs to be done differently for multi sub array jigsaw handling
        # move to when making the mask and final stitching together different pieces to do the border
        self.master_dark[:, :4, :] = 0.
        self.master_dark[:, -4:, :] = 0.
        self.master_dark[:, :, :4] = 0.
        self.master_dark[:, :, -4:] = 0.
        logging.info(f'Master dark attribute created.')

    def save_master_dark(self, md_outfile=None):
        """
        The method save_master_dark with default conditions will write the master dark cube into an asdf
        file for each detector in the directory from which the input files where pointed to and used to
        construct the master dark read cube. A user can specify an absolute path or relative file string
        to write the master dark file name to disk.

        Parameters
        ----------
        md_outfile: str; default = None
            File string. Absolute or relative path for optional input.
            By default, None is provided but the method below generates the asdf file string from meta
            data of the input files such as date and detector number  (i.e. WFI01) in the filename.
        """

        meta_md = {'pedigree': "GROUND", 'description': "Master dark internal reference file calibration product"
                                                        "generated from Reference File Pipeline.",
                   'date': Time(datetime.datetime.now()), 'detector': self.meta['instrument']['detector']}
        if md_outfile is None:
            md_outfile = os.path.dirname(self.input_data[0]) + '/' + meta_md['detector'] + '_master_dark.asdf'
        self.check_output_file(md_outfile)
        logging.info(f'Saving master dark to disk.')

        af = asdf.AsdfFile()
        af.tree = {'meta': meta_md, 'data': self.master_dark}
        af.write_to(md_outfile)

    def make_ma_table_sequence(self, read_pattern):
        """
        Function to generate a flattened list of integer indices representing reads in each resultant that
        are separated by the character 'R" to denote the reads composing of a resultant.
        Example: ma_table_sequence = [1,R,3,4,R] is an exposure where the first resultant is composed of the first
        read only, with read 2 being a skip, and reads 3 and 4 averaged together in the second resultant.

        Parameters
        ----------
        read_pattern: list
            Nested list of integer lists representing reads per resultant in exposure.
        """

        rds_per_res_list = list(map(np.shape, read_pattern))
        for res in read_pattern:
            for rd in res:
                self.ma_table_sequence.append(rd)
            self.ma_table_sequence.append('R')

    def make_ma_table_dark(self, num_resultants, num_rds_per_res=None):
        """
        The method make_dark() takes a non-resampled dark cube read and converts it into
        a number of resultants that constructed from the mean of a number of reads
        as specified by the MA table ID. The number of reads per resultant,
        the number of resultants, and the MA table ID are inputs to creating
        the resampled dark cube.

        NOTE: Future work will have the MA table ID as input and internally
        reference the RTB Database to retrieve MA table properties (i.e. the
        number of reads per resultant and number of resultants, and possible
        sequence of reads to achieve unevenly spaced resultants. Currently
        assuming equally spaced resultants.

        Parameters
        ----------
        num_resultants: integer
            The number of resultants
        num_rds_per_res: integer; Default=None
            The user supplied number of reads per resultant in evenly spaced resultants.
        """

        # Flow control and logging messaging depending on how the Dark() class is initialized
        if self.input_data is not None and self.dark_read_cube is None:
            self.dark_read_cube = self.master_dark
            logging.info(f'Master dark created from input filelist used for MA table resampling.')
        if self.dark_read_cube is not None:
            logging.info(f'Input dark read cube used for MA table resampling.')
        self.n_reads, self.ni, _ = np.shape(self.dark_read_cube)

        # Make the time array for the length of the dark read cube exposure.
        self.make_time_array()

        # Initialize resampled dark cube, error, and time arrays with number of ma tables specs
        self.resampled_dark_cube = np.zeros((num_resultants, self.ni, self.ni), dtype=np.float32)
        self.resampled_dark_cube_err = np.zeros((num_resultants, self.ni, self.ni), dtype=np.float32)
        logging.info(f'Error arrays with number of resultants initialized with zeros.')
        logging.info(f'Run calc_dark_err_metrics() to calculate errors.')
        self.resultant_tau_arr = np.zeros(num_resultants, dtype=np.float32)

        # Perform evenly spaced sampling if the keyword num_rds_per_res is supplied and it has an integer value
        if self.ma_table_sequence is not None:
            print(self.ma_table_sequence[:])
            msg = f'Using MA table exposure sequence generated with make_ma_table_sequence method() for instructions' \
                  f'on MA table resampling and averaging of resultants.'
            logging.info(msg)
            print(msg)
            # For unevenly spaced resultant time tau in Casternao et al equation 14 handling the variance based
            # resultant time
        else:
            if num_rds_per_res is None:
                msg = 'Not enough information provided to do MA table resampling. Provide num_resultants with keyword' \
                      ' num_rds_per_res to perform even resampling averaging or provide ma_table_seq'
                logging.info(msg)
                raise ValueError(msg)
            if num_rds_per_res > self.n_reads:
                # Check that the length of the reads per resultant is not greater than the available number of reads
                logging.info(f'Can not average over more reads than supplied in dark read cube.')
                raise ValueError(f'Can not average over more reads than supplied in dark read cube.')
            # Averaging over reads per ma table specs or user defined even spacing.
            logging.info(f'Averaging over reads with evenly spaced resultants.')
            for i_res in range(0, num_resultants):
                i1 = i_res * num_rds_per_res
                i2 = i1 + num_rds_per_res
                if i2 > self.n_reads:
                    logging.info(f'Warning: The number of reads per resultant was not evenly divisible into the number'
                                 f' of available reads to average and remainder reads were skipped.')
                    logging.info(f'Resultants after resultant {i_res+1} contain zeros.')
                    break
                self.resampled_dark_cube[i_res, :, :] = np.mean(self.dark_read_cube[i1:i2, :, :], axis=0)
            self.resultant_tau_arr[i_res] = np.mean(self.time_arr[i1:i2])
            logging.info(f'MA table resampling with {num_resultants} resultants averaging {num_rds_per_res}'
                         f' reads per resultant complete.')

    def make_time_array(self):
        """
        The method make_data_time_arrays() will generate a WFI mode dependent time array of the exposure or input
        data supplied to make the read noise reference file.
        """

        if self.frame_time is None:
            if self.meta.p_exptype == 'WFI_IMAGE':
                self.frame_time = self.ancillary['frame_time']['WIM']  # frame time in imaging mode in seconds
            elif self.meta.p_exptype == 'WFI_GRISM':
                self.frame_time = self.ancillary['frame_time']['WSM']  # frame time in spectral mode in seconds
            else:
                logging.info(f'No frame time found for WFI mode specified.')
                raise ValueError(f'No frame time found for WFI mode specified!')

        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.n_reads} reads long with a frame'
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.n_reads + 1)])

    def calc_dark_err_metrics(self, hot_pixel_rate=0.015, warm_pixel_rate=0.010, hot_pixel_bit=11, warm_pixel_bit=12):
        """
        The calc_dark_err_metrics() method computes the error as the variance of the fitted ramp or slope
        along the time axis for the resultants in the resampled_dark_cube attribute using a 1st order polyfit.
        The slopes are saved as the dark_rate_image and the variances as the dark_rate_image_var. The hot and warm
        pixel thresholds are applied to the dark_rate_image and the pixels are identified with their respective
        DQ bit flag.

        NOTE: Look into writing metrics like number of hot pixels here or in a different method altogether. The
        determination of hot and warm pixel rates and sigma values are only referenced as a best guess
        at what we should consider for setting these values. Likely to change or be more informed post-TVAC.

        Parameters
        ----------
        hot_pixel_rate: float; default = 0.015 DN/s or ADU/s
            The hot pixel rate is the number of DN/s determined from detector characterization to be 10-sigma above
            the nominal expectation of dark current.
        warm_pixel_rate: float; default = 0.010 e/s
            The warm pixel rate is the number of DN/s determined from detector characterization to be 8-sigma above
            the nominal expectation of dark current.
        hot_pixel_bit: integer; default = 11
            DQ hot pixel flag value in romancal library.
        warm_pixel_bit: integer; default = 12
            DQ hot pixel flag value in romancal library.
        """

        logging.info(f'Computing dark image variance and noise.')
        # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.
        num_resultants, _, _ = np.shape(self.resampled_dark_cube)
        p, c = np.polyfit(self.resultant_tau_arr,
                          self.resampled_dark_cube.reshape(len(self.resultant_tau_arr), -1), 1, full=False, cov=True)

        # Reshape results back to 2D arrays.
        dark_rate_image = p[0].reshape(self.ni, self.ni)  # the fitted ramp slope image
        dark_intercept_image = p[1].reshape(self.ni, self.ni)  # the fitted y intercept
        dark_rate_var = c[0, 0, :].reshape(self.ni, self.ni)  # covariance matrix slope variance
        dark_intercept_var = c[1, 1, :].reshape(self.ni, self.ni)  # covariance matrix intercept variance

        # Generate a dark ramp cube model per the resampled ma table specs.
        self.resampled_dark_cube_model = np.zeros((len(self.resampled_dark_cube), self.ni, self.ni), dtype=np.float32)
        for tt in range(0, len(self.resultant_tau_arr)):
            self.resampled_dark_cube_model[tt, :, :] = dark_rate_image * self.resultant_tau_arr[tt] \
                                                       + dark_intercept_image  # y = m*x + b
        # Calculate the residuals of the dark ramp model and the data
        residual_cube = self.resampled_dark_cube_model - self.resampled_dark_cube
        std = np.std(residual_cube, axis=0)  # this is the standard deviation of residuals from the resampled dark cube
        # model and the resampled dark cube data, std^2 is therefore the resampled read noise variance
        # the dark cube error array should be a 2D image of 4096x4096 with the slope variance from the model fit
        # and the variance of the resampled residuals are added in quadrature
        self.resampled_dark_cube_err[0, :, :] = (std * std + dark_rate_var) ** 0.5

        logging.info(f'Flagging hot and warm pixels and updating DQ array.')
        # Locate hot and warm pixel ni,nj positions in 2D array
        hot_pixels = np.where(dark_rate_image >= hot_pixel_rate)
        warm_pixels = np.where((warm_pixel_rate <= dark_rate_image) & (dark_rate_image < hot_pixel_rate))

        # Set mask DQ flag values
        self.mask[hot_pixels] += 2 ** hot_pixel_bit
        self.mask[warm_pixels] += 2 ** warm_pixel_bit

        # Get the number of hot and warm pixels for metric tracking
        _, num_hot_pixels = np.shape(hot_pixels)
        _, num_warm_pixels = np.shape(warm_pixels)
        logging.info(f'Found {num_hot_pixels} hot pixels and {num_warm_pixels} warm pixels in dark rate ramp image.')

    def save_dark(self):
        """
        The method save_dark() writes the resampled dark cube into an asdf file to be saved somewhere on disk.
        Read
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)
        logging.info(f'Writing new dark reference file to disk.')

        # Construct the dark object from the data model.
        dark_file = rds.DarkRef()
        dark_file['meta'] = self.meta
        dark_file['data'] = self.resampled_dark_cube * ru.DN
        dark_file['err'] = self.resampled_dark_cube_err * ru.DN
        dark_file['dq'] = self.mask

        # af: asdf file tree: {meta, data, err, dq}
        af = asdf.AsdfFile()
        af.tree = {'roman': dark_file}
        af.write_to(self.outfile)
