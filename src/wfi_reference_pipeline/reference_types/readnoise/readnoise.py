import roman_datamodels.stnode as rds
from astropy import units as u
import numpy as np
import asdf
import logging
import math
import gc
import os
from astropy.stats import sigma_clip
from ..reference_type import ReferenceType
from wfi_reference_pipeline.resources.wfi_meta_readnoise import WFIMetaReadNoise
from wfi_reference_pipeline.constants import WFI_MODE_WIM, WFI_MODE_WSM, WFI_TYPE_IMAGE, WFI_FRAME_TIME


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
                 input_file_list,
                 meta_data=None,
                 bit_mask=None,
                 outfile='roman_readnoise.asdf',
                 clobber=False,
                 input_data_cube=None):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        input_file_list: string object; default = None
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
            The first positional variable in the ReadNoise class instance assigned in base class ReferenceType().
            For ReadNoise() self.input_data is a list of string filenames with paths.

        See ReferenceType() base class for additional attributes available to all reference file types such
        as ancillary data.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            input_file_list,
            meta_data,
            bit_mask=bit_mask,
            clobber=clobber,
            make_mask=True
        )

        if not isinstance(meta_data, WFIMetaReadNoise):
            raise TypeError(f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaReadNoise")
        if len(self.meta.description) == 0:
            self.meta.description = 'Roman WFI read noise reference file.'

        #TODO Would feedback on how many attributes I have and if they seem logical and organized properly
        # Initialize attributes
        self.outfile = outfile
        # Additional object attributes
        # Inputs
        self.input_data_cube = input_data_cube  # Default to user supplied input data cube, over write if filelist
        # Internal
        self.ramp_model = None  # Ramp model fitted to a data cube.
        self.ramp_res_var = None  # The variance of residuals of the difference between the ramp model and a
        # data cube.
        self.cds_noise = None  # The correlated double sampling noise estimate between successive pairs of reads
        # in a data cube.
        # Readnoise attributes
        self.readnoise_image = None

        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_reads = None  # Number of reads in data cube being analyzed.
        self.ni = None  # Number of pixels.
        self.frame_time = None  # Mode dependent exposure frame time per read.
        self.time_arr = None  # Time array of an exposure or input data.

        #TODO input data into the module through the base class should ALWAYS be a list of files
        # Additional inputs are needed also but not each module is the same or will have the same number or types of additional inputs
        #TODO can we standardize the flow here such that if files are provided run under normal automated operations and
        # when provided other additional input it uses that.
        #TODO Think of user from public github wanting to input different data - files vs cubes
        # Check input data to instantiated ReadNoise().
        if self.input_data is None and self.input_data_cube is None:
            raise ValueError('No data supplied to make read noise reference file!')
        if self.input_data is not None and len(self.input_data) > 0 and \
                self.input_data_cube is not None and len(self.input_data_cube) > 0:
            raise ValueError('Two inputs provided. Not sure how to proceed. Provide files or a data cube; not both.')

    def make_readnoise_image(self):
        """
        This method determines the flow of the module based on the input data
        when the class is instantiated. The read noise rate image is created at the
        end of the method to be the data in the datamodel.
        """

        #TODO is should be the generic type method in each reference file type that is called after the class is instantiated
        # Utilize nested methods from most complex starting with files to simplest where an array is sent to make the file

        # Use input files if they exist.
        if self.input_data is not None:
            print('Making READNOISE from input files')
            logging.info('Using file list to make make read noise.')
            self._select_data_cube()
            self.readnoise_image = self.comp_ramp_res_var()
        # Attempt to use input data cube.
        try:
            shape = self.input_data_cube.shape
            if len(shape) == 2:
                logging.info('Input data is a 2D array. Writing array to self.readnoise_image.')
                self.readnoise_image = self.input_data_cube
            if len(shape) == 3:
                n_reads = shape[0]
                logging.info(f'Input data cube has {n_reads} reads.')
                self.readnoise_image = self.comp_ramp_res_var()
        except Exception as e:
            raise ValueError('Input data is not a compatible numpy array shape.') from e

    def _select_data_cube(self):
        """
        The method select_data_cube() looks through the file list provided to ReadNoise() and finds the file with
        the most number of reads. It sorts the files in descending order by the number of reads such that the
        first index will be the file with the most number of reads and the last will have the fewest.

        #TODO This could be useful to compare cube with varying numbers of reads and the calculated read noise.
        """

        logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to find file longest exposure'
                     f'and the most number of reads.')
        # Go through all files to sort them from the longest to shortest number of reads available.
        fl_reads_ordered_list = []
        for fl in range(0, len(self.input_data)):   # TODO - can the n_reads be in the DAAPI metadata so we can avoid opening each file just to get that?
            with asdf.open(self.input_data[fl]) as tmp:
                n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
                fl_reads_ordered_list.append([self.input_data[fl], n_rds])
        # Sort the list of files in reverse order such that the file with the most number of reads is always in
        # the zero index first element of the list.
        fl_reads_ordered_list.sort(key=lambda x: x[1], reverse=True)

        # Get the input file with the most number of reads from the sorted list.
        with asdf.open(fl_reads_ordered_list[0][0]) as tmp:
            self.input_data_cube = tmp.tree['roman']['data']
        logging.info(f'Using the file {fl_reads_ordered_list[0][0]} to get a read noise cube.')

    def _initialize_arrays(self):
        """
        Method initialize_arrays makes arrays of the dimensions of the dark_read_cube, which are also required
        in the data model.
        """

        self.n_reads, self.ni, _ = np.shape(self.input_data_cube)

        # Make the time array for the length of the dark read cube exposure.
        if self.meta.type == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WSM]  # frame time in spectral mode in seconds
        # Generate the time array depending on WFI mode.
        logging.info(f'Creating exposure time array {self.n_reads} reads long with a frame '
                     f'time of {self.frame_time} seconds.')
        self.time_arr = np.array([self.frame_time * i for i in range(1, self.n_reads + 1)])

    def _make_ramp_cube_model(self):
        """
        Method make_ramp_cube_model performs a linear fit to the input read cube for each pixel. The slope
        and intercept are returned along with the covariance matrix which has the corresponding diagonal error
        estimates for variances in the model fitted parameters.
        """

        #TODO fitting a ramp to a cube of data is probably common to all modules
        #TODO making a ramp model is probably common to all modules

        logging.info('Making ramp model for the input read cube.')

        # Reshape the 2D array into a 1D array for input into np.polyfit(). The model fit parameters p and
        # covariance matrix v are returned.
        # TODO - This will currently blow up with some test data, try except, or
        p, v = np.polyfit(self.time_arr,
                          self.input_data_cube.reshape(len(self.time_arr), -1), 1, full=False, cov=True)

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
        residual_cube = self.ramp_model - self.input_data_cube.value      # TODO - THIS BREAKS!
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

        Parameters
        ----------
        sig_clip_cds_low: float; default = 5.0
            Lower bound limit to filter difference cube.
        sig_clip_cds_high: float; default = 5.0
            Upper bound limit to filter difference cube
        """

        #TODO this wants to be a method accessible to a user to produce the readnoise reference files
        # If this is selected, comp_ramp_res_var should not be available

        logging.info('Calculating CDS noise.')

        self._initialize_arrays()
        self._make_ramp_cube_model()

        read_diff_cube = np.zeros((math.ceil(self.n_reads / 2), self.ni, self.ni), dtype=np.float32)
        for i_read in range(0, self.n_reads - 1, 2):
            # Avoid index error if n_reads is odd and disregard the last read because it does not form a pair.
            logging.info(f'Calculating correlated double sampling between frames {i_read} and {i_read + 1}')
            rd1 = self.ramp_model[i_read, :, :] - self.input_data_cube[i_read, :, :]
            rd2 = self.ramp_model[i_read + 1, :, :] - self.input_data_cube[i_read + 1, :, :]
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1
        clipped_diff_cube = sigma_clip(read_diff_cube, sigma_lower=sig_clip_cds_low, sigma_upper=sig_clip_cds_high,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
        self.cds_noise = np.std(clipped_diff_cube, axis=0)

        #TODO keeping track of memory resources might be a good thing to do for each module. I'd like to be lean
        del read_diff_cube
        gc.collect()

        return self.cds_noise

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the read noise object from the data model.
        readnoise_datamodel_tree = rds.ReadnoiseRef()
        readnoise_datamodel_tree['meta'] = self.meta.export_asdf_meta()
        readnoise_datamodel_tree['data'] = self.readnoise_image * u.DN

        return readnoise_datamodel_tree

    def save_readnoise(self, datamodel_tree=None):
        """
        The method save_readnoise writes the reference file object to the specified asdf outfile.
        """

        #TODO if we made an abstract method in base class that appended save_ with REFTYPE that might be slick
        # but also probably hard to do but this is the same code in each module

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
