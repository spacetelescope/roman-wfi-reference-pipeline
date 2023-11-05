import roman_datamodels.stnode as rds
from astropy import units as u
import numpy as np
import asdf
import logging
import math
import gc
import os
from astropy.stats import sigma_clip
from ..utilities.logging_functions import configure_logging
from ..utilities.reference_file import ReferenceFile
from wfi_reference_pipeline.constants import WFI_MODE_WIM, WFI_MODE_WSM, WFI_TYPE_IMAGE, WFI_FRAME_TIME
configure_logging('readnoise_dev', path='/grp/roman/RFP/DEV/logs/')


class ReadNoise(ReferenceFile):
    """
    Class ReadNoise() inherits the ReferenceFile() base class methods where static meta data for all reference
    file types are written. Under automated operational conditions, a dark calibration file with the most number
    of reads for each detector will be selected from a list of dark calibration input files from the input data variable
    in ReferenceFile() and ReadNoise(). Dark calibration files where every read is available and not averaged are the
    best available data to measure the variance of the detector read by read. A ramp model for all available reads
    will be subtracted from the input data cube that is constructed from the input file list provided and the variance
    in the residuals is determined to be the best measurement of the read noise (Casterano and Cosentino email
    discussions Dec 2022).

    Additional complexity such as the treatment of Poisson noise, shot noise, read-out noise, etc. are
    to be determined. The method get_cds_noise() is available for diagnostics purposes and comparison when developing
    more mature functionality of the reference file pipeline.
    """

    def __init__(self, input_file_list, meta_data=None, bit_mask=None, outfile=None, clobber=False,
                 input_data_cube=None):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------
        input_filelist: string object; default = None
            List of dark calibration filenames with absolute paths. If no file list is provided, an input dark read cube
            should be supplied.
        meta_data: dictionary; default = None
            Dictionary of information for read noise reference file as required by romandatamodels.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer array for supplying a mask for the creation of the read noise reference file.
        outfile: string; default = roman_readnoise.asdf
            Filename with path for saved read noise reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        input_data_cube: numpy array; default = None
            Data cube of reads. Dimensions of n_reads x ni x ni, where ni is the number of pixels of a square array
            of the detector by the number of reads (n_reads). NOTE - For parallelization only square arrays allowed.
        ----------
        self.input_data: attribute;
            The first positional variable in the ReadNoise class instance assigned in base class ReferenceFile().
            For ReadNoise() self.input_data is a list of string filenames with paths.

        See ReferenceFile() base class for additional attributes available to all reference file types such
        as ancillary data.
        """

        # Access methods of base class ReferenceFile().
        super(ReadNoise, self).__init__(input_file_list, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with read noise file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI read noise reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'READNOISE'
        else:
            pass

        # If no output filename given, set default filename.
        self.outfile = outfile if outfile else 'roman_read_noise.asdf'
        logging.info(f'Default read noise reference file object: {outfile} ')

        # Additional object attributes
        self.input_filelist_cube = None  # Cube of data generated from input file list
        self.input_readnoise_cube = input_data_cube  # Default to user supplied input data cube, over write if filelist
        self.ramp_model = None  # Ramp model fitted to input data cube.
        self.ramp_res_var = None  # The variance in the residuals of the difference between the ramp model and input
        # data cube that was created from one of the dark calibration files or the input_read_cube supplied.
        self.cds_noise = None  # The correlated double sampling noise estimate between successive pairs of reads/frames.
        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_reads = None  # Number of reads in data cube being analyzed.
        self.ni = None  # Number of pixels.
        self.frame_time = None  # Mode dependent exposure frame time per read.
        self.time_arr = None  # Time array of an exposure or input data.

        # Check input data to initialize ReadNoise().
        if self.input_data is None and self.input_readnoise_cube is None:
            raise ValueError('No data supplied to make read noise reference file!')

    def make_read_cube(self):
        """
        The method make_read_cube() looks through the filelist provided to the ReadNoise module and finds the file with
        the most number of reads in that list. It currently sorts the files in descending order of the number
        of reads such that the first index will be the file with the most number of reads and the last will have the
        fewest. This functionality could be useful for looking at how many reads are necessary to accurately quantify
        the read noise.
        """

        # Determine what type of input data was passed. If input data is not none, then go through the list of files.
        if self.input_data is not None and self.input_readnoise_cube is None:
            logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to find file longest exposure'
                         f'and the most number of reads.')
            # Go through all dark files to sort them from the longest to shortest number of reads available.
            fl_reads_ordered_list = []
            for fl in range(0, len(self.input_data)):
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
                fl_reads_ordered_list.append([self.input_data[fl],n_rds])
                tmp.close()
            # Sort the list of files in reverse order such that the file with the most number of reads is always in
            # the zero index first element of the list.
            fl_reads_ordered_list.sort(key=lambda x: x[1], reverse=True)

            # Get the input dark file with the most number of reads from the sorted list.
            tmp = asdf.open(fl_reads_ordered_list[0][0], validate_on_read=False)
            self.input_filelist_cube = tmp.tree['roman']['data']
            logging.info(f'Using the file {fl_reads_ordered_list[0][0]} to get a read noise cube.')

        elif self.input_data is None and self.input_readnoise_cube is not None:
            logging.info('No files to make read cube. User supplied input read cube.')
            raise ValueError('User supplied input read cube. No action taken by method make_read_cube()!')
        else:
            raise ValueError('Expected either a file list or an input data read cube. Not both!')

    def initialize_arrays(self):
        """
        Method initialize_arrays makes arrays of the dimensions of the dark_read_cube, which are also required
        in the data model.

        Parameters
        ----------
        ni: integer; Default=None
            Number of square pixels of array ni.
        """

        if self.input_filelist_cube:
            self.input_readnoise_cube = self.input_filelist_cube
            self.n_reads, self.ni, _ = np.shape(self.input_readnoise_cube)
            logging.info('')
        elif self.input_readnoise_cube:
            self.n_reads, self.ni, _ = np.shape(self.input_readnoise_cube)
        else:
            raise ValueError('Expected either an input file list cube or a user supplied input data cube.')

        # Initialize ramp residual variance array.
        self.ramp_res_var = np.zeros((self.ni, self.ni), dtype=np.float32)

        # Make the time array for the length of the dark read cube exposure.
        if self.meta['exposure']['type'] == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WSM]  # frame time in spectral mode in seconds
        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.n_reads} reads long with a frame '
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.n_reads + 1)])

    def make_ramp_cube_model(self):
        """
        Method make_ramp_cube_model performs a linear fit to the input read cube for each pixel. The slope
        and intercept are returned along with the covariance matrix which has the corresponding diagonal error
        estimates for variances in the model fitted parameters.
        """

        logging.info('Making ramp model for the input read cube.')

        # Reshape the 2D array into a 1D array for input into np.polyfit(). The model fit parameters p and
        # covariance matrix v are returned.
        p, v = np.polyfit(self.time_arr, self.input_readnoise_cube.reshape(len(self.time_arr), -1), 1, full=False, cov=True)

        # Reshape the parameter slope array into a 2D rate image.
        ramp_image = p[0].reshape(self.ni, self.ni)
        # Reshape the parameter y-intercept array into a 2D image.
        intercept_image = p[1].reshape(self.ni, self.ni)
        # Reshape the returned covariance matrix slope fit error.
        # ramp_var = v[0, 0, :].reshape(self.ni, self.ni) TODO -VERIFY USE
        # returned covariance matrix intercept error.
        # intercept_var = v[1, 1, :].reshape(self.ni, self.ni) TODO - VERIFY USE

        self.ramp_model = np.zeros((self.n_reads, self.ni, self.ni), dtype=np.float32)
        for tt in range(0, len(self.time_arr)):
            # Construct a simple linear model y = m*x + b.
            self.ramp_model[tt, :, :] = ramp_image * self.time_arr[tt] + intercept_image

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

        logging.info('Computing residuals of ramp model from data to estimate variance component of read noise.')

        residual_cube = self.ramp_model - self.input_readnoise_cube
        clipped_res_cube = sigma_clip(residual_cube, sigma_lower=sig_clip_res_low, sigma_upper=sig_clip_res_high,
                                      cenfunc=np.mean, axis=0, masked=False, copy=False)
        std = np.std(clipped_res_cube, axis=0)
        self.ramp_res_var = np.float32(std*std)

    def comp_cds_noise(self, sig_clip_cds_low=5.0, sig_clip_cds_high=5.0):
        """
        Compute the correlated double sampling as a noise estimate. The method get_cds_noise() calculates the
        correlated double sampling between pairs of reads in the data cube as a noise term from the standard deviation
        of the differences from all read pairs.

        Parameters
        ----------
        sig_clip_cds_low: float; default = 5.0
            Lower bound limit to filter difference cube.
        sig_clip_cds_high: float; default = 5.0
            Upper bound limit to filter difference cube
        """

        logging.info('Calculating CDS noise.')

        read_diff_cube = np.zeros((math.ceil(self.n_reads / 2), self.ni, self.ni), dtype=np.float32)
        for i_read in range(0, self.n_reads - 1, 2):
            # Avoid index error if n_reads is odd and disregard the last read because it does not form a pair.
            logging.info(f'Calculating correlated double sampling between frames {i_read} and {i_read + 1}')
            rd1 = self.ramp_model[i_read, :, :] - self.input_readnoise_cube[i_read, :, :]
            rd2 = self.ramp_model[i_read + 1, :, :] - self.input_readnoise_cube[i_read + 1, :, :]
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1
        clipped_diff_cube = sigma_clip(read_diff_cube, sigma_lower=sig_clip_cds_low, sigma_upper=sig_clip_cds_high,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
        self.cds_noise = np.std(clipped_diff_cube, axis=0)

        del read_diff_cube
        gc.collect()

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the read noise object from the data model.
        readnoise_datamodel_tree = rds.ReadnoiseRef()
        readnoise_datamodel_tree['meta'] = self.meta
        readnoise_datamodel_tree['data'] = self.ramp_res_var * u.DN

        return readnoise_datamodel_tree

    def save_readnoise(self, datamodel_tree=None):
        """
        The method save_readnoise writes the reference file object to the specified asdf outfile.
        """

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
