import roman_datamodels.stnode as rds
import numpy as np
import psutil, sys, os, glob, time, gc, asdf, math, datetime, logging
from ..utilities.reference_file import ReferenceFile
from ..utilities.logging_functions import configure_logging
from astropy.stats import sigma_clip
from astropy.time import Time
from RTB_Database.utilities.login import connect_server
from RTB_Database.utilities.table_tools import DatabaseTable
from RTB_Database.utilities.table_tools import table_names

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

    def __init__(self, dark_filelist, meta_data=None, bit_mask=None, outfile=None, clobber=False,
                 input_dark_cube=None, wfi_mode=None):
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
        dark_read_cube: numpy array; default = None
            Cube of dark reads to be resampled into MA table specific dark reference file. Dimensions of
            ni x ni x n_reads, where ni is the number of pixels of a square sub-array of the detector by the number of
            reads (n_reads) in the integration. NOTE - For parallelization only square arrays allowed.
        wfi_mode: string; default = None
            WFI imaging (WIM) or WFI spectral (WSM) modes. The indicated mode is used to generate the time sequence
            array which is dependent on mode specific integration time.
        -------
        self.input_data: variable;
            The first positional variable in the Dark class instance assigned in base class ReferenceFile().
            For Dark() self.input_data is a list of string filenames with paths.
        """

        # Access methods of base class ReferenceFile
        super(Dark, self).__init__(dark_filelist, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=True)

        # Update metadata with dark file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI dark reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'DARK'
        else:
            pass

        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_dark.asdf'
        logging.info(f'Default dark reference file object: {outfile} ')

        # Additional object attributes
        self.dark_read_cube = input_dark_cube   # Supplied input dark read cube.
        self.wfi_mode = wfi_mode  # WFI Imaging (WIM) or Spectral (WSM) mode.
        self.master_dark = None  # A cube of all available reads that is sigma clipped and averaged.
        self.ma_table_seq = None  # For general treatment of unevenly spaced resultant averaging.
        self.resampled_dark_cube = None  # MA table averaged resultant cube.
        self.resampled_dark_cube_model = None  # MA table resultant ramp model.
        self.resampled_dark_cube_err = None  # MA table averaged resultant error cube.
        self.resultant_tau_arr = None  # Variance-based resultant time tau_i from Casterano et al. 2022 equation 14.
        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_reads = None  # Number of reads in data cube being analyzed.
        self.ni = None  # Number of pixels.
        self.exp_time = None  # Frame time from ancillary data.
        self.time_arr = None  # Time array of an exposure.

        if self.input_data is None and self.dark_read_cube is None:
            raise ValueError(f'No data supplied to make dark reference file!')

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

    def get_ma_table_info(self, ma_table_id):
        """
        This method get_me_table_info() imports modules and methods from the RTB database
        repo to allow the RFP to establish a connection and query the science ma tables
        for specifications on how they are made.

        NOTE: Incorporating changes from the upscope are not included yet and will alter
        how this is done in significant manners.

        It might be that this is also a utility in order to initiliaze dark reference file
        meta data with ma table information before the Dark class() instance is created

        Parameters
        ----------
        ma_table_id: integer
            Database entry in first column to fetch ma table information.

        Returns
        -------
        ma_tab_num_resultants: integer
            number of resultants, currently called ngroups in meta
        ma_tab_reads_per_resultant: integer
            number of reads per resultant, currently called nframe in meta
        """

        con, _, _ = connect_server(DSN_name='DWRINSDB')
        new_tab = DatabaseTable(con, 'ma_table_science')
        ma_tab = new_tab.read_table()
        ma_tab_ind = ma_table_id - 1  # to match index starting at 0 in database with integer ma table ID starting at 1
        ma_tab_name = ma_tab.at[ma_tab_ind, 'ma_table_name']
        ma_tab_reads_per_resultant = ma_tab.at[ma_tab_ind, 'read_frames_per_resultant']
        ma_tab_num_resultants = ma_tab.at[ma_tab_ind, 'resultant_frames_onboard']
        self.exp_time = ma_tab.at[ma_tab_ind, 'detector_read_time']
        ma_tab_reset_read_time = ma_tab.at[ma_tab_ind, 'detector_reset_read_time']
        logging.info(f'Retrieved RTB Database multi-accumulation (MA) table ID {ma_table_id}.')
        logging.info(f'MA table {ma_tab_name} has {ma_tab_num_resultants} resultants and {ma_tab_reads_per_resultant}'
                     f' reads per resultant.')
        # now update meta data with ma table specs
        self.meta['exposure'].update(dict(ngroups=ma_tab_num_resultants, nframes=ma_tab_reads_per_resultant, groupgap=0,
                                          ma_table_name=ma_tab_name, ma_table_number=ma_table_id))
        logging.info(f'Updated meta data with MA table info.')

        # Determine WIM or WSM - WFI Imaging or Spectral Mode - from the ma table specs in the RTB database from PRD.
        if self.exp_time == self.ancillary['frame_time']['WIM']:  # frame time in imaging mode in seconds
            self.meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
            logging.info(f'WFI spectral mode selected (WIM).')
        elif self.exp_time == self.ancillary['frame_time']['WSM']:  # frame time in spectral mode in seconds:
            self.meta['exposure'].update({'type': 'WFI_GRISM', 'p_exptype': 'WFI_GRISM|WFI_PRISM|'})
            logging.info(f'WFI spectral mode selected (WSM).')

        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array in {self.wfi_mode} mode with {self.n_reads} reads with a frame'
                     f'time of {self.exp_time} seconds.')
        self.time_arr = np.array([self.exp_time * i for i in range(1, self.n_reads + 1)])

        return ma_tab_num_resultants, ma_tab_reads_per_resultant

        # Separate method here to make MA table time array, exposure sequence and where a user supplied MA table
        # name and mode and sequence can be used to to make a dark without connection to the RTB database.

    def make_ma_table_dark(self, num_resultants, reads_per_resultant):
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
        num_resultants: integer; the number of resultants
            The number of final resultants in the dark asdf file.
        reads_per_resultant: integer; the number of reads per resultant
            The number of reads to be averaged in creating a resultant.
        """

        # Flow control and logging messaging depending on how the Dark() class is initialized
        if self.input_data is not None and self.dark_read_cube is None:
            self.dark_read_cube = self.master_dark
            logging.info(f'Master dark created from input filelist used for MA table resampling.')
        if self.dark_read_cube is not None:
            logging.info(f'Input dark read cube used for MA table resampling.')
        self.n_reads, self.ni, _ = np.shape(self.dark_read_cube)

        # Initialize resampled dark cube, error, and time arrays with number of ma tables specs
        self.resampled_dark_cube = np.zeros((num_resultants, self.ni, self.ni), dtype=np.float32)
        self.resampled_dark_cube_err = np.zeros((num_resultants, self.ni, self.ni), dtype=np.float32)
        logging.info(f'Error arrays with number of resultants initialized with zeros.')
        logging.info(f'Run calc_dark_err_metrics() to calculate errors.')
        self.resultant_tau_arr = np.zeros(num_resultants, dtype=np.float32)

        # Averaging over reads per ma table specs
        for i_res in range(0, num_resultants):
            i1 = i_res * reads_per_resultant
            i2 = i1 + reads_per_resultant
            if i2 > self.n_reads:
                break
            self.resampled_dark_cube[i_res, :, :] = np.mean(self.dark_read_cube[i1:i2, :, :], axis=0)

            # NOTE: Below needs to be updated for the generalized unevenly spaced resultant time tau in Casternao et al
            # from equation 14 handling the variance-based resultant time - tau_i
            # if evenly spaced should just be the mid point average of read times composing the resultant
            self.resultant_tau_arr[i_res] = np.mean(self.time_arr[i1:i2])

        logging.info(f'MA table resampling with {num_resultants} resultants averaging {reads_per_resultant}'
                     f' reads per resultant complete.')
        logging.info(f'Updated meta data with MA Table information.')

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
        self.resampled_dark_cube_err[0, :, :] = (std*std + dark_rate_var)**0.5

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
        dark_file['data'] = self.resampled_dark_cube
        dark_file['err'] = self.resampled_dark_cube_err
        dark_file['dq'] = self.mask

        # af: asdf file tree: {meta, data, err, dq}
        af = asdf.AsdfFile()
        af.tree = {'roman': dark_file}
        af.write_to(self.outfile)