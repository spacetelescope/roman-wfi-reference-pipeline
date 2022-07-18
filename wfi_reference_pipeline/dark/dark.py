import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf, logging
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
import pandas as pd
import psutil, sys, os, glob, time, gc

# Squash logging messages from stpipe.
logging.getLogger('stpipe').setLevel(logging.WARNING)

class Dark(ReferenceFile):
    """
    Class Dark() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_dark() implements specified MA table properties (number of
    reads per resultant). The dark asdf file has contains the averaged dark
    frame resultants.
    """

    def __init__(self, dark_read_cube=None, meta_data=None, bit_mask=None, clobber=False, outfile=None,
                 dark_filelist=None, master_dark=None, resampled_dark_cube=None, resampled_dark_cube_err=None):

        # Access methods of base class ReferenceFile
        super(Dark, self).__init__(dark_read_cube, meta_data, bit_mask=bit_mask, clobber=clobber)

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

        # Object attributes for master dark
        self.dark_filelist = dark_filelist
        self.master_dark = master_dark
        # Object attributes for resampled darks
        self.resampled_dark_cube = resampled_dark_cube
        self.resampled_dark_cube_err = resampled_dark_cube_err

    def make_master_dark(self, sigma_clip_low_bound=3.0, sigma_clip_high_bound=3.0, ma_table_id=None):
        """
        The method make_master_dark() ingests all files located in a directory as a python object list of
        file names with absolute path. A master dark is created by iterating through each read of every
        dark file, read by read (see NOTE below). A cube of reads is formed into a numpy array and sigma
        clipped according to the input attributes and the mean of the clipped data cube is saved into
        the master dark array.

        NOTE: The process above is file I/O intensive but utilizes less memory while performance is only
        marginally slower. Initial testing was performed by R. Cosentino with 12 dark files that each had
        21 reads where this method took ~330 seconds and had a peak memory usage of 2.5 GB. Opening and loading
        all files at once took ~300 seconds with a peak memory usage of 36 GB. Running from the command line
        or in the ipython intrepreter displays significant difference in memory and run time.

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
        mem2 = psutil.Process().memory_info().rss / (1024. * 1024.)  # physical memory in MB
        #print('Initial memory usage = {:.1f} MB'.format(mem2))
        #print(self.dark_filelist)
        #print("Method 2 reads the dimensions of one file and then makes master dark which is computed by"
              #"opening each file read by read, sigma clipping, and then averaging. This appears to use"
              #"less memory than method 1 while taking slightly longer in time. Need to test this with more"
              #"TVAC and operations based darks with number of reads per file and number of files.")

        start = time.time()
        ftmp = asdf.open(self.dark_filelist[0], validate_on_read=False)
        n_reads, ni, nj = np.shape(ftmp.tree['roman']['data'])
        # need to look through files and check dimensions and if not all the same
        # raise exception to use what is there to make master dark
        n_files = len(self.dark_filelist)
        self.master_dark = np.zeros((n_reads, 4096, 4096), dtype=np.float32)
        for rd in range(0, n_reads):
            dark_read_cube = []
            for fl in range(0, n_files):
                ftmp = asdf.open(self.dark_filelist[fl], validate_on_read=False)
                rd_tmp = ftmp.tree['roman']['data']
                dark_read_cube.append(rd_tmp[rd, :, :])
                mem2 = psutil.Process().memory_info().rss / (1024. * 1024.)  # physical memory in MB
                #print('Memory used in file loop', mem2, ' MB')
                del ftmp, rd_tmp
                gc.collect()
            clipped_reads = sigma_clip(dark_read_cube, sigma_lower=sigma_clip_low_bound, sigma_upper=sigma_clip_high_bound,
                                       cenfunc=np.mean, axis=0, masked=False, copy=False)
            self.master_dark[rd, :, :] = np.mean(clipped_reads, axis=0)
            mem2 = psutil.Process().memory_info().rss / (1024. * 1024.)  # physical memory in MB
            #print('Memory used in read loop', mem2, ' MB')
            del clipped_reads
            gc.collect()

        end = time.time()
        #print('Method 2 took', end - start, ' seconds.')
        mem2 = psutil.Process().memory_info().rss / (1024. * 1024.)  # physical memory in MB
        #print('Final memory usage = {:.1f} MB'.format(mem2))

        # set reference pixel border to zero for master dark
        self.master_dark[:, :4, :] = 0.
        self.master_dark[:, -4:, :] = 0.
        self.master_dark[:, :, :4] = 0.
        self.master_dark[:, :, -4:] = 0.

    def make_dark(self, num_resultants, reads_per_resultant, ma_table_ID):
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
        ma_table_ID: integer; the MA table ID number 1-999
            The MA table name ID number is used to retrieve table parameters
            for converting the dark read cube into resultants and averages
            used during science observations

        Returns
        -------
        None.
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # initialize dark cube with number of resultants from inputs
        self.resampled_dark_cube = np.zeros((num_resultants, 4096, 4096), dtype=np.float32)
        self.resampled_dark_cube_err = np.zeros((num_resultants, 4096, 4096), dtype=np.float32)

        # average over number of reads per resultant for each pixel
        # to make darks according to ma_table specs
        for i_res in range(0,num_resultants):
            i1 = i_res * reads_per_resultant
            i2 = i1 + reads_per_resultant
            self.resampled_dark_cube[i_res, :, :] = np.mean(self.data[i1:i2, :, :], axis=0)
            res_std = np.std(self.resampled_dark_cube[i_res, :, :])
            self.resampled_dark_cube_err[i_res, :, :] = np.round(np.random.uniform(.9*res_std, 1.1*res_std,
                                                                    size=(4096, 4096)).astype(np.float32),2)

        # log info of darks made
        #logging.info('Darks were made with %s resultants made with %s reads per resultant',
        #             str(n_result), str(n_read_per_result))

    def compute_dark_metrics(self):
        # hot pixel flagging will require
        fresult_avg = np.mean(self.data[-1])
        hot_pixels = np.where(self.data[-1] > 1.8*fresult_avg)
        self.mask[hot_pixels] = 2 ** 11
        num_hotpixels = np.count_nonzero(self.mask == 2 **11)

    def save_dark_file(self):
        """
        The method save_dark_file() writes the resampled dark cube into an asdf
        file to be saved somewhere on disk.

        Returns
        -------
        af: asdf file tree: {meta, data, dq, err}
            meta:
            data: averaged resultants per MA table specs
            dq: mask - data quality array
                masked hot pixels in rate image flagged 2**11
            err: zeros
        """
        dark_file = rds.DarkRef()
        dark_file['data'] = self.resampled_dark_cube
        dark_file['err'] = self.resampled_dark_cube_err
        dark_file['dq'] = self.mask
        dark_file['meta'] = self.meta
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': dark_file}
        af.write_to(self.outfile)



