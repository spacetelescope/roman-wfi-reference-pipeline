import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import psutil, sys, os, glob, time, gc, asdf, logging, math
import numpy as np
from astropy.stats import sigma_clip
from RTB_Database.utilities.login import connect_server
from RTB_Database.utilities.table_tools import DatabaseTable
from RTB_Database.utilities.table_tools import table_names

# Squash logging messages from stpipe.
logging.getLogger('stpipe').setLevel(logging.WARNING)
log_file_str = 'dark_dev.log'
logging.basicConfig(filename=log_file_str, level=logging.INFO)
logging.info(f'Dark reference file log: {log_file_str}')


class Dark(ReferenceFile):
    """
    Class Dark() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_dark() implements specified MA table properties (number of
    reads per resultant). The dark asdf file has contains the averaged dark
    frame resultants.
    """

    def __init__(self, dark_filelist, meta_data=None, bit_mask=None, clobber=False, outfile=None,
                 dark_read_cube=None):
        """
        The __init__ method initializes the class with proper data nedded to be sent to the ReferenceFile
        file base class. The general input data sent to ReferenceFile by the Dark class is a list of filename
        string objects. Meta_data is populated or modified at certain points in the dark reference file creation
        such as the number of resultants in an MA table. Outfile is the output filename and path to be written
        to disk.

        Parameters
        ----------
        dark_filelist: string object; default = None
            List of dark filenames with absolute paths. If no filelist provided, an input dark read cube
            should be supplied for MA table resampling to create dark reference file.
        meta_data: dictionary; default = None
            Dictionary of information for dark reference file as required by romandatamodels
        outfile: string; default = None
            Outfile defined as roman_dark.asdf if no outfile given and Dark class save_dark() method is executed.
        dark_read_cube: numpy array; default = None
            Cube of dark reads to be resampled into MA table specific dark reference file. Dimensions
            of ni x ni x n_reads, where ni is the number of pixels of a square array of the detector by the
            number of reads (n_reads) in the integration.  NOTE: For parallelization only square arrays allowed.
            dimensions of dark_read_cube = [ni,ni,n_reads]

        Returns
        -------
        None
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

        # Object attributes for master dark
        self.master_dark = None
        # Object attributes for resampled darks
        self.dark_read_cube = dark_read_cube
        self.resampled_dark_cube = None
        self.resampled_dark_cube_err = None

    def make_master_dark(self, sigma_clip_low_bound=3.0, sigma_clip_high_bound=3.0, write_md=1):
        """
        The method make_master_dark() ingests all files located in a directory as a python object list of
        file names with absolute path. A master dark is created by iterating through each read of every
        dark file, read by read (see NOTE below). A cube of reads is formed into a numpy array and sigma
        clipped according to method inputs (default is 3 sigma) and the mean of the clipped data cube is
        saved as the master dark class attribute.

        NOTE: The algorithm is file I/O intensive but utilizes less memory while performance is only currently
        only marginally slower. Initial testing was performed by R. Cosentino with 12 dark files that each had
        21 reads where this method took ~330 seconds and had a peak memory usage of 2.5 GB. Opening and loading
        all files at once took ~300 seconds with a peak memory usage of 36 GB. Running from the command line
        or in the ipython intrepreter displays significant differences in memory usage/allocation and run time.

        Parameters
        ----------
        sigma_clip_low_bound: float; default = 3.0
            Lower bound limit to filter data.
        sigma_clip_high_bound: float; default = 3.0
            Upper bound limit to filter data

        Returns
        -------
        None
        """

        logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to construct master dark object.')

        # for next phase of development query what file types are in directory, HDF5, fits, asdf,
        # then cycle through to find longest number of reads to initialize master dark and have all
        # data converted into a numpy array
        tmp_reads = []
        for fl in range(0, len(self.input_data)):
            ftmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_reads, ni, _ = np.shape(ftmp.tree['roman']['data'])
            tmp_reads.append(n_reads)
        num_reads_set = [*set(tmp_reads)]

        # The master dark length is the maximum number of reads in all dark asdf files to be used
        # when creating the dark reference file. Need to "try" over files with different lengths
        # to compute average read by read for all files
        self.master_dark = np.zeros((np.max(num_reads_set), ni, ni), dtype=np.float32)
        # This method of opening and close each file read by read is file I/O intensive however
        # it is efficient on memory usage. Compare memory and time for stress tests.
        logging.info(f'Reading dark asdf files read by read to compute average for master dark.')
        for rd in range(0, np.max(num_reads_set)):
            dark_read_cube = []
            logging.info(f'On read {rd} of {np.max(num_reads_set)}')
            for fl in range(0, len(self.input_data)):
                ftmp = asdf.open(self.input_data[fl], validate_on_read=False)
                rd_tmp = ftmp.tree['roman']['data']
                dark_read_cube.append(rd_tmp[rd, :, :])
                del ftmp, rd_tmp
                gc.collect()  # clean up memory
            clipped_reads = sigma_clip(dark_read_cube, sigma_lower=sigma_clip_low_bound,
                                       sigma_upper=sigma_clip_high_bound,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
            self.master_dark[rd, :, :] = np.mean(clipped_reads, axis=0)
            del clipped_reads
            gc.collect()  # clean up memory

        # write master dark to disk for use by other modules
        # or as a means for module development to load a numpy array that is the master dark
        # attribute but does not have to run the make_master_dark() method

        if write_md == 1:
            master_dark_fl_str = 'master_dark_cube.dat'
            master_dark_dat_fl = os.path.dirname(self.input_data[0]) + '/' + master_dark_fl_str
            fp = np.memmap(master_dark_dat_fl, dtype='float32', mode='w+', shape=(n_reads, ni, ni))
            logging.info(f'Master dark cube written to file in directory as: {master_dark_fl_str}')

        # set reference pixel border to zero for master dark
        # this needs to be done differently for multi sub array jigsaw handling
        # move to when making the mask and final stitching together different pieces to do the border
        self.master_dark[:, :4, :] = 0.
        self.master_dark[:, -4:, :] = 0.
        self.master_dark[:, :, :4] = 0.
        self.master_dark[:, :, -4:] = 0.
        logging.info(f'Master dark attribute created.')

    def get_ma_table_info(self, ma_table_ID):
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
        ma_table_ID: integer
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
        ma_tab_ind = ma_table_ID - 1  # to match index starting at 0 in database with integer ma table ID starting at 1
        ma_tab_name = ma_tab.at[ma_tab_ind, 'ma_table_name']
        ma_tab_reads_per_resultant = ma_tab.at[ma_tab_ind, 'read_frames_per_resultant']
        ma_tab_num_resultants = ma_tab.at[ma_tab_ind, 'resultant_frames_onboard']
        ma_tab_read_time = ma_tab.at[ma_tab_ind, 'detector_read_time']
        ma_tab_reset_read_time = ma_tab.at[ma_tab_ind, 'detector_reset_read_time']
        logging.info(f'Retrieved RTB Database multi-accumulation (MA) table ID {ma_table_ID}.')
        logging.info(f'MA table {ma_tab_name} has {ma_tab_num_resultants} resultants and {ma_tab_reads_per_resultant}'
                     f' reads per resultant.')
        # generate time array for exposure sequence for ramp fitting and error and variance calcs
        # first instance of attribute in Dark class
        self.exp_time_arr = [ma_tab_read_time * i for i in
                             range(1, ma_tab_reads_per_resultant * ma_tab_num_resultants + 1)]
        logging.info(f'Constructed exposure time array sequence from MA table information on read spacing, skips,'
                     f'and number of resultants')

        # check if we should write this meta data here or update the dictionary key values
        self.meta['exposure'] = {'ngroups': ma_tab_num_resultants, 'nframes': ma_tab_reads_per_resultant,
                                 'groupgap': 0, 'ma_table_name': ma_tab_name, 'ma_table_number': ma_table_ID,
                                 'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'}
        logging.info(f'Updated meta data with MA table info.')

        return ma_tab_num_resultants, ma_tab_reads_per_resultant

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
        ma_table_ID: integer; the MA table ID number 1-999
            The MA table name ID number is used to retrieve ma table parameters
            for converting the master dark read cube into resultant  averages
            used during science observations
        num_resultants: integer; the number of resultants
            The number of final resultants in the dark asdf file.
        reads_per_resultant: integer; the number of reads per resultant
            The number of reads to be averaged in creating a resultant.

        Returns
        -------
        None.
        """

        # control and messaging depending on how the Dark() class is initialized
        if self.dark_read_cube is not None:
            logging.info(f'Input dark read cube used for MA table resampling.')
        if self.input_data is not None and self.dark_read_cube is None:
            self.dark_read_cube = self.master_dark
            logging.info(f'Master dark created from input filelist used for MA table resampling.')
        n_reads, ni, _ = np.shape(self.dark_read_cube)

        # initialize dark cube, error, and time arrays with number of resultants from inputs
        self.resampled_dark_cube = np.zeros((num_resultants, ni, ni), dtype=np.float32)
        self.resampled_dark_cube_err = np.zeros((num_resultants, ni, ni), dtype=np.float32)
        self.resampled_dark_time_arr = np.zeros((num_resultants), dtype=np.float32)

        # average over number of reads per resultant per ma table specs
        for i_res in range(0, num_resultants):
            i1 = i_res * reads_per_resultant
            i2 = i1 + reads_per_resultant
            if i2 > n_reads:
                break
            self.resampled_dark_cube[i_res, :, :] = np.mean(self.dark_read_cube[i1:i2, :, :], axis=0)
            # to handle upscope with variance-based resultant time - tau_i in uneven spaced resultant
            # if evenly spaced, equation 14 reduces to below
            self.resampled_dark_time_arr[i_res] = np.mean(self.exp_time_arr[i1:i2])
            # to do next phase, implement equation 14 here

        logging.info(f'MA table resampling with {num_resultants} resultants averaging {reads_per_resultant}'
                     f' reads per resultant complete.')
        logging.info(f'Updated meta data with MA Table information.')
        logging.info(f'Error arrays initialized with zeros. Run calc_dark_err_metrics() to calculate errors.')

    def calc_dark_err_metrics(self, hot_pixel_rate=0.015, warm_pixel_rate=0.010,
                              hot_pixel_bit=11, warm_pixel_bit=12):
        """
        The calc_dark_err_metrics() method computes the error as the variance of the fitted ramp or slope
        along the time axis for the resultants in the resampled_dark_cube attribute using a 1st order polyfit.
        The slopes are saved as the dark_rate_image and the variances as the dark_rate_image_var. The hot and warm
        pixel thresholds are applied to the dark_rate_image and the pixels are identified with their respective
        DQ bit flag.

        NOTE: Look into writing metrics like number of hot pixels here or in a different method altogether

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

        NOTE: The determination of hot and warm pixel rates and sigma values are only referenced as a best guess
        at what we should consider for setting these values. Likely to change or be more informed post-TVAC.
        """

        logging.info(f'Computing dark variance and CDS noise values.')
        # linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency
        num_resultants, ni, _ = np.shape(self.resampled_dark_cube)
        slopes, variances = np.polyfit(self.resampled_dark_time_arr,
                                       self.resampled_dark_cube.reshape(len(self.resampled_dark_time_arr), -1), 1)

        # reshape results back to 2D arrays
        self.dark_rate_image = slopes.reshape(ni, ni)
        self.dark_rate_image_var = variances.reshape(ni, ni)

        # calculate correlated double sampling between pairs of successive reads - the read noise
        # consider making this its own module in RFP utilities - CDS_noise() module and method
        n_reads, ni, _ = np.shape(self.dark_read_cube)
        read_diff_cube = np.zeros((math.ceil(n_reads / 2), ni, ni), dtype=np.float32)
        for i_read in range(0, n_reads-1, 2):
            # n_reads-1 skip last pair to avoid index error if n_reads is odd
            logging.info(f'Calculating correlated double sampling between frames {i_read} and {i_read+1}')
            rd1_ramp_sub = self.dark_read_cube[i_read, :, :] - self.dark_rate_image * self.exp_time_arr[i_read]
            rd2_ramp_sub = self.dark_read_cube[i_read + 1, :, :] - self.dark_rate_image * self.exp_time_arr[i_read + 1]
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2_ramp_sub - rd1_ramp_sub
        # A. Petric noted using 3 or 5 sigma clipping to compute CDS noise, consider for next phase
        cds_noise = np.std(read_diff_cube[:, :, :], axis=0)
        del read_diff_cube
        gc.collect()

        # add errors from slope fitting and read noise in quadrature
        # next step is to implement general error measurements of variance in unevenly spaced resultant and slope
        # fitting memo
        for i_result in range(0,num_resultants):
            self.resampled_dark_cube_err[i_result, :, :] = (cds_noise**2 + self.dark_rate_image_var**2)**0.5

        logging.info(f'Flagging hot and warm pixels and updating DQ array.')
        # locate hot and warm pixel ni,nj positions in 2D array
        hot_pixels = np.where(self.dark_rate_image >= hot_pixel_rate)
        warm_pixels = np.where((warm_pixel_rate <= self.dark_rate_image) & (self.dark_rate_image < hot_pixel_rate))

        # set mask DQ flag values
        self.mask[hot_pixels] += 2 ** hot_pixel_bit
        self.mask[warm_pixels] += 2 ** warm_pixel_bit

        # get the number of hot and warm pixels for metric tracking
        _, num_hot_pixels = np.shape(hot_pixels)
        _, num_warm_pixels = np.shape(warm_pixels)
        logging.info(f'Found {num_hot_pixels} hot pixels and {num_warm_pixels} warm pixels in dark rate ramp image.')

    def save_dark(self):
        """
        The method save_dark_file() writes the resampled dark cube into an asdf
        file to be saved somewhere on disk.

        NOTE: Consider this method to stitch chunks of jigsaw approach of subarrays back together
        into the dimensions for writing the final version of the asdf reference file.

        Returns
        -------

        af: asdf file tree: {meta, data, dq, err}
            meta:
            data: averaged resultants per MA table specs
            dq: mask - data quality array
                masked hot pixels in rate image flagged 2**11
            err: zeros
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        dark_file = rds.DarkRef()
        dark_file['data'] = self.resampled_dark_cube
        dark_file['err'] = self.resampled_dark_cube_err
        dark_file['dq'] = self.mask
        dark_file['meta'] = self.meta
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': dark_file}
        af.write_to(self.outfile)
