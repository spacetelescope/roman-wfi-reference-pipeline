import logging
import math
import os

import asdf
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u
from astropy.stats import sigma_clip
from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_MODE_WSM,
    WFI_TYPE_IMAGE,
)
from wfi_reference_pipeline.resources.wfi_meta_readnoise import WFIMetaReadNoise

from ..reference_type import ReferenceType


class ReadNoise(ReferenceType):
    """
    Class ReadNoise() inherits the ReferenceType() base class methods where static meta data for all reference
    file types are written. Under automated operational conditions, a dark calibration file with the most number
    of reads for each detector will be selected from a list of dark calibration input files from the input data variable
    in ReferenceType() and ReadNoise(). Dark calibration files where every read is available and not averaged are the
    best available data to measure the variance of the detector read by read. A ramp model for all available reads
    will be subtracted from the input data cube that is constructed from the input file list provided and the variance
    in the residuals is determined to be the best measurement of the read noise (Casterano and Cosentino email
    discussions Dec 2022).

    Additional complexity such as the treatment of Poisson noise, shot noise, read-out noise, etc. are
    to be determined. The method get_cds_noise() is available for diagnostics purposes and comparison when developing
    more mature functionality of the reference file pipeline.
    """

    def __init__(self,
                 meta_data,
                 file_list=None,
                 data_array=None,
                 bit_mask=None,
                 outfile='roman_readnoise.asdf',
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

        # Inherit reference_type.
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            data_array=data_array,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber,
            make_mask=True
        )

        if not isinstance(meta_data, WFIMetaReadNoise):
            raise TypeError(f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaReadNoise")
        if len(self.meta_data.description) == 0:
            self.meta_data.description = 'Roman WFI read noise reference file.'

        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.n_files = len(self.file_list)
            self._select_data_cube_from_file_list()
            # Must make_readnoise_image() to finish creating reference file.
        else:
            # Get data array properties.
            dim = self.data_array.shape
            if isinstance(self.data_array, u.Quantity):  # Only access data from quantity object.
                self.data_array = self.data_array.value
            if len(dim) == 2:
                logging.info('The input 2D data array is now self.readnoise_image.')
                self.readnoise_image = self.data_array
                logging.info('Ready to generate reference file.')
            elif len(dim) == 3:
                logging.info('User supplied 3D data cube to make read noise reference file.')
                self.data_cube = self.data_array
                self.readnoise_image = None  # Making read noise from data cube
                # Must make_readnoise_image() to finish creating reference file.
            else:
                raise ValueError('Input data is not a valid numpy array of dimension 2 or 3.')

        self.ramp_res_var = None  # The variance of residuals from the difference of the ramp model and a data cube.
        self.cds_noise = None  # The correlated double sampling noise estimate between successive pairs
        # of reads in data cube.

        #TODO data cube class
        #self.data_cube = None  # Data cube processed by methods to make read noise image.
        self.ni = None  # Number of pixels.
        self.n_reads = None  # Number of reads in data.
        self.ramp_model = None  # Ramp model of data cube.
        self.frame_time = None  # Mode dependent exposure frame time per read.
        self.time_arr = None  # Time array of data cube.

    def make_readnoise_image(self):
        """
        This method is used to generate the reference file image type from the file list or a data cube.
        """

        logging.info('Making read noise image.')
        self.readnoise_image = self.comp_ramp_res_var()

    def _select_data_cube_from_file_list(self):
        """
        The method select_data_cube() looks through the file list provided to ReadNoise() and finds the file with
        the most number of reads. It sorts the files in descending order by the number of reads such that the
        first index will be the file with the most number of reads and the last will have the fewest.
        """

        logging.info(f'Using files from {os.path.dirname(self.file_list[0])} to find file longest exposure'
                     f'and the most number of reads.')
        # Go through all files to sort them from the longest to shortest number of reads available.
        fl_reads_ordered_list = []
        for fl in range(0, len(self.file_list)):
            #TODO update using rdm.open() method
            with asdf.open(self.file_list[fl]) as tmp:
                n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
                fl_reads_ordered_list.append([self.file_list[fl], n_rds])
        # Sort the list of files in reverse order such that the file with the most number of reads is always in
        # the zero index first element of the list.
        fl_reads_ordered_list.sort(key=lambda x: x[1], reverse=True)

        # Get the input file with the most number of reads from the sorted list.
        #TODO update using rdm.open() method
        with asdf.open(fl_reads_ordered_list[0][0]) as tmp:
            self.data_cube = tmp.tree['roman']['data']
            if isinstance(self.data_cube, u.Quantity):  # Only access data from quantity object.
                self.data_cube = self.data_cube.value

        logging.info(f'Using the file {fl_reads_ordered_list[0][0]} to get a read noise cube.')

    def _initialize_arrays(self):
        """
        # TODO data cube class
        Get dimensions of self.data_cube.
        Initialize time array after getting frame time from meta.
        Initialize empty ramp_model.
        """


        self.n_reads, self.ni, _ = np.shape(self.data_cube)

        # Make the time array for the length of the dark read cube exposure.
        if self.meta_data.type == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WSM]  # frame time in spectral mode in seconds
        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.n_reads} reads long with a frame '
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.n_reads + 1)])

    def _make_ramp_cube_model(self):
        """
        # TODO data cube class
        Change this to fit ramp model with optional input for order of polyfit with default order = 1
        Return or save to attribute the ramp image and intercept image.
        Keep covariance matrices in code for future use determination.

        Method make_ramp_cube_model performs a linear fit to the input read cube for each pixel. The slope
        and intercept are returned along with the covariance matrix which has the corresponding diagonal error
        estimates for variances in the model fitted parameters.
        """


        logging.info('Making ramp model for the input read cube.')

        # Reshape the 2D array into a 1D array for input into np.polyfit(). The model fit parameters p and
        # covariance matrix v are returned.

        try:
            p, v = np.polyfit(self.time_arr,
                              self.data_cube.reshape(len(self.time_arr), -1), 1, full=False, cov=True)
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

        except (ValueError, TypeError) as e:
            logging.error(f"Unable to make_ramp_cube_model with error {e}")
            # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, I ASSUME WE CAN'T JUST LOG IT - For cube class discussion

    #TODO default parameter values for accessible methods
    # What about inaccessible methods?
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

        #TODO this wants to be a method accessible to a user to produce the readnoise reference files
        # If this is selected, comp_cds_noise should not be available

        logging.info('Computing residuals of ramp model from data to estimate variance component of read noise.')

        self._initialize_arrays()
        self._make_ramp_cube_model()

        # Initialize ramp residual variance array.
        self.ramp_res_var = np.zeros((self.ni, self.ni), dtype=np.float32)
        residual_cube = self.ramp_model - self.data_cube
        clipped_res_cube = sigma_clip(residual_cube, sigma_lower=sig_clip_res_low, sigma_upper=sig_clip_res_high,
                                      cenfunc=np.mean, axis=0, masked=False, copy=False)
        std = np.std(clipped_res_cube, axis=0)
        self.ramp_res_var = np.float32(std * std)
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

        # If this is selected, comp_ramp_res_var should not be available

        logging.info('Calculating CDS noise.')

        self._initialize_arrays()
        self._make_ramp_cube_model()

        read_diff_cube = np.zeros((math.ceil(self.n_reads / 2), self.ni, self.ni), dtype=np.float32)
        for i_read in range(0, self.n_reads - 1, 2):
            # Avoid index error if n_reads is odd and disregard the last read because it does not form a pair.
            logging.info(f'Calculating correlated double sampling between frames {i_read} and {i_read + 1}')
            rd1 = self.ramp_model[i_read, :, :] - self.data_cube[i_read, :, :]
            rd2 = self.ramp_model[i_read + 1, :, :] - self.data_cube[i_read + 1, :, :]
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1
        clipped_diff_cube = sigma_clip(read_diff_cube, sigma_lower=sig_clip_cds_low, sigma_upper=sig_clip_cds_high,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
        self.cds_noise = np.std(clipped_diff_cube, axis=0)

        return self.cds_noise

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the read noise object from the data model.
        readnoise_datamodel_tree = rds.ReadnoiseRef()
        readnoise_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        readnoise_datamodel_tree['data'] = self.readnoise_image.astype(np.float32) * u.DN

        return readnoise_datamodel_tree