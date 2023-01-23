import roman_datamodels.stnode as rds
import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf, logging, math, gc, os
from ..utilities.logging_functions import configure_logging
import numpy as np
from astropy.stats import sigma_clipped_stats

configure_logging('readnoise_dev', path='/grp/roman/RFP/DEV/logs/')
#configure_logging('readnoise_dev')


class ReadNoise(ReferenceFile):
    """
    Class ReadNoise() inherits the ReferenceFile() base class methods where static meta data for all reference
    file types are written. Under automated operations conditions, a dark read cube with the most amount of reads
    for a particular detector and mode will be selected from a list of input files that is the input data from
    ReferenceFile() into ReadNoise(). Since every dark read is available and not averaged, this is the best available
    data to measure variance read by read. A ramp model for all available reads will be subtracted from the input
    data and the variance in the residuals is determined to be the best measurement of the read noise.

    Additional complexity may be added. Treatment of Poisson noise, shot noise, read-out noise, etc. are TBD.

    Additional methods, such as the get_cds_noise() method, are available for diagnostics and comparison when
    developing more mature functionality into the reference file pipeline.
    """

    def __init__(self, input_filelist, meta_data=None, bit_mask=None, outfile=None, clobber=False, wfi_mode='IMAGE',
                 input_read_cube=None):
        """
        The __init__ method initializes the class with proper data needed to be sent to the ReferenceFile
        file base class. The general input data sent to ReferenceFile by the ReadNoise class is a read cube of
        data of a given length for which noise will be measured over. In general, this should be the master dark
        cube from the Dark() module or a user supplied. Meta_data from the master dark might be used in lieu of
        the key word parameter wfi_mode which is necessary to determine the exposure read time. Outfile is the
        output filename and path to be written to disk.

        Parameters
        ----------
        input_filelist: string object; default = None
            List of dark filenames with absolute paths. If no filelist provided, an input dark read cube
            should be supplied.
        meta_data: dictionary; default = None
            Dictionary of information for read noise reference file as required by romandatamodels.
        outfile: string; default = None
            Outfile defined as roman_readnoise.asdf if no outfile given and ReadNoise class save_read_noise() method
            is executed.
        wfi_mode: string; default = 'IMAGE'
            The mode for which the WFI was in and consistent with the input_data. WFI imaging and spectral modes have
            different integration times and therefore the time sequence generated from the input data is determined by
            which mode was used to acquire the data.
        input_read_cube: numpy array; default = None
            Cube of dark reads. Dimensions of ni x ni x n_reads, where ni is the number of pixels of a square sub-array
            of the detector by the number of reads (n_reads) in the integration. NOTE: For parallelization only square
            arrays allowed.

        Returns
        -------
        None
        """

        # Access methods of base class ReferenceFile
        super(ReadNoise, self).__init__(input_filelist, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with read noise file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI read noise reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'READNOISE'
        else:
            pass

        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_read_noise.asdf'
        logging.info(f'Default read noise reference file object: {outfile} ')

        # other object attributes
        self.ramp_res_var = None  # the variance in the residuals from the difference of the a ramp model and input data
        self.cds_noise = None  # the correlated double sampling estimate between successive pairs of reads/frames
        self.WFI_MODE = wfi_mode
        self.input_read_cube = input_read_cube

    def get_read_cube(self):
        """
        The method get_read_cube() looks through the filelist provided to the ReadNoise module and finds the data
        set with the most number of reads in that list. It currently sorts the files in descending order of the number
        of reads such that the first index will be the longest number of reads and the last will be the fewest. This
        functionality could be useful for looking at how many reads are necessary to accurately determine read noise.
        The number of reads is used to construct a time sequence for the integration of whatever file has the most
        reads. The mode for WFI is used as selector criteria for selecting the appropriate exposure time in seconds.

        Parameters
        ----------

        Returns
        -------
        None
        """

        # Determine what type of input data was passed - either a list of files or a data array.
        if self.input_data is not None and self.input_read_cube is None:
            logging.info(f'Using files from {os.path.dirname(self.input_data[0])} to find longest input file exposure.')

            # go through all dark files to find longest amount of reads available
            fl_reads_ordered_list = []
            for fl in range(0, len(self.input_data)):
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                n_rds, _, _ = np.shape(tmp.tree['roman']['data'])
                fl_reads_ordered_list.append([self.input_data[fl],n_rds])
                tmp.close()
            fl_reads_ordered_list.sort(key=lambda x: x[1], reverse=True)

            # get the input file with the most reads from the sorted list
            tmp = asdf.open(fl_reads_ordered_list[0][0], validate_on_read=False)
            self.input_read_cube = tmp.tree['roman']['data']
            logging.info(f'Using {fl_reads_ordered_list[0][0]} to compute noise.')

        self.n_reads, self.ni, _ = np.shape(self.input_read_cube)
        if self.WFI_MODE == 'SPECTRAL':
            self.exp_time = self.ancillary['frame_time']['WSM']  # frame time in spectral mode in seconds
        elif self.WFI_MODE == 'IMAGE':
            self.exp_time = self.ancillary['frame_time']['WIM']  # frame time in imaging mode in seconds

        # generate the time array depending on which WFI mode is determined from above
        logging.info(f'Creating exposure integration for {self.WFI_MODE} mode with {self.n_reads} reads with a frame'
                     f'time of {self.exp_time} seconds.')
        self.time_seq = np.array([self.exp_time * i for i in range(1, self.n_reads + 1)])

    def make_ramp_cube_model(self):
        """
        Method make_ramp_cube_model performs a linear fit to the data in the input data cube for each pixel. The slope
        and intercept are returned as well as the covariance matrix which has the corresponding diagonal error
        estimates which may be used in later analysis.

        NOTE: A. Petric and Calibration Block that this method is very similar for how the read noise component in the
        Dark module is computed. The difference here in ReadNoise is that EVERY READ IS USED IN THE READ NOISE as the
        variance of the residuals from the model and the data, while in Dark, ONLY THE MA TABLE RESAMPLED NUMBER OF
        READS ARE USED.

        Returns
        -------
        None
        """

        logging.info(f'Making ramp model for the input read cube.')

        p, V = np.polyfit(self.time_seq, self.input_read_cube.reshape(len(self.time_seq), -1), 1, full=False, cov=True)

        # reshape results back to 2D arrays
        self.ramp_image = p[0].reshape(self.ni, self.ni)  # the ramp slope fit
        self.intercept_image = p[1].reshape(self.ni, self.ni)  # the y intercept of the slope
        #self.ramp_var = V[0, 0, :].reshape(ni, ni)  # returned covariance matrix slope fit error
        #self.intercept_var = V[1, 1, :].reshape(ni, ni)  # returned covariance matrix intercept error

        self.ramp_model = np.zeros((len(self.input_read_cube), self.ni, self.ni), dtype=np.float32)
        for tt in range(0, len(self.time_seq)):
            self.ramp_model[tt, :, :] = self.ramp_image * self.time_seq[tt] + self.intercept_image  # y = m*x + b

    def get_ramp_res_var(self):
        """
        Method get_ramp_res_var() uses the difference of the data from a ramp model for each pixel as the residuals
        for which the variance of that cube is the most appropriate method for computing the read noise.

        NOTE: A. Petric and Calibration Block that this method is very similar for how the read noise component in the
        Dark module is computed. The difference here in ReadNoise is that EVERY READ IS USED IN THE READ NOISE as the
        variance of the residuals from the model and the data, while in Dark, ONLY THE MA TABLE RESAMPLED NUMBER OF
        READS ARE USED.

        Returns
        -------
        None
        """

        logging.info(f'Computing residuals of ramp model from data to estimate variance component of read noise.')

        residual_cube = self.ramp_model - self.input_read_cube
        std = np.std(residual_cube, axis=0)
        self.ramp_res_var = std*std

    def get_cds_noise(self):
        """
        Method get_cds_noise() calculates the correlated double sampling between pairs of reads in the data cube.
        This noise is computed from the standard deviation of the difference of all read pairs.

        Returns
        -------
        None
        """

        logging.info(f'Calculating CDS noise.')

        read_diff_cube = np.zeros((math.ceil(self.n_reads / 2), self.ni, self.ni), dtype=np.float32)
        for i_read in range(0, self.n_reads - 1, 2):
            # n_reads-1 skip last pair to avoid index error if n_reads is odd
            logging.info(f'Calculating correlated double sampling between frames {i_read} and {i_read + 1}')
            rd1 = self.ramp_model[i_read, :, :] - self.input_read_cube[i_read, :, :]
            rd2 = self.ramp_model[i_read + 1, :, :] - self.input_read_cube[i_read + 1, :, :]
            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1  # differnce of reads
        # NOTE that A. Petric use 3 or 5 sigma clipping to compute CDS noise, consider for next phase
        self.cds_noise = np.std(read_diff_cube[:, :, :], axis=0)
        del read_diff_cube
        gc.collect()

    def save_read_noise(self):
        """
        The method save_read_noise() writes the read noise cube into an asdf
        file to be saved somewhere on disk.

        Returns
        -------
        af: asdf file tree: {meta, data, dq, err}
            meta:
            data: read noise array
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the read noise object from the data model.
        rn_file = rds.ReadnoiseRef()
        rn_file['meta'] = self.meta
        rn_file['data'] = self.ramp_res_var
        # NOTE: read noise files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': rn_file}
        af.write_to(self.outfile)