import datetime
import gc
import logging
import os
import time
import asdf
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy import units as u
from pathlib import Path
import roman_datamodels as rdm
import psutil


def get_mem_usage():
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)  # in GB
    return memory_usage


class SuperDark:
    """
    SuperDark() is a class that will ingest raw L1 dark calibration files and average every read for
    as many exposures as there are available to create a superdark.asdf file. This file is the assumed input
    into the Dark() module in the RFP and the input used for resampling to make dark calibration
    reference files for any given number of MA Tables.
    """

    def __init__(
        self,
        input_path,
        file_list,
        outfile="roman_superdark.asdf",
    ):
        """
        The __init__ method initializes the class.

        Parameters
        ----------
        input_path: str,
            Path to input directory where files are located.
        file_list: list,
            List of files in the input_directory
        outfile: str, default="roman_superdark.asdf"
            File name written to disk.
        """

        self.input_path = Path(input_path)
        self.file_list = file_list
        self.outfile = outfile
        self.superdark = None
        self.meta_data = None
        self.max_reads = None
        self.n_reads_list = None

        self.pre_avg_cube = None
        self.flxfl_read_avg_cube = None

    def get_file_list_meta_rdmopen(self):
        """
        This method uses rdm.open() to get all of the meta relevant in generating the super dark.

        NOTE: Future Consideration - This method may be deemed unnecessary given information from DAAPI
        during RFP queries to get data.

        NOTE: Processing Time and Memory - This method takes around 500 seconds on average to process
        opening and extracting meta from the 50 files for WFI01 according to the in-flight calibration
        program from Casertano et al. 2022. Since each file is opened here on;y for meta data, time is
        saved when generating the super dark since not every file needs to be opened and on the fly
        checked for the number of reads and if its data should be included.

        """

        file_name_list = []  # empty filename list
        n_reads_list = []  # empty number of reads in file list
        exp_time_start_list = []  # empty exposure start time list
        exp_type_list = []  # empty exposure type list

        print("Testing meta information retrieval")
        print(f"Memory used at start of get_file_list_meta_rdmopen: {get_mem_usage():.2f} GB")
        timing_start_getmeta = time.time()

        logging.info("Testing meta information retrieval")
        logging.info(f"Memory used at start of get_file_list_meta_rdmopen: {get_mem_usage():.2f} GB")

        for file_name in self.file_list:
            file_path = self.input_path.joinpath(file_name)
            logging.info(f"Opening file {file_path}")
            try:
                with rdm.open(file_path) as af:
                    meta = af.meta
                    filename = meta.filename
                    read_pattern = meta.exposure.read_pattern
                    exp_time_start = meta.exposure.start_time
                    exp_type = meta.exposure.type
                del af
                gc.collect()
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                print(f"Error opening file {file_path}: {e}")

            # Process each file's metadata here
            # TODO figure out what we need from each file
            # print(f"File: {meta.filename}")
            # print(f"Read Pattern: {meta.exposure.read_pattern}")
            # print(f"Start Time: {meta.exposure.start_time}")
            # print(f"Exposure Type: {meta.exposure.type}")
            # print(f"Memory used in file_name loop after opening file: {get_mem_usage():.2f} GB")
            logging.info(f"File: {meta.filename}")
            logging.info(f"Read Pattern: {meta.exposure.read_pattern}")
            logging.info(f"Start Time: {meta.exposure.start_time}")
            logging.info(f"Exposure Type: {meta.exposure.type}")

            file_name_list.append(filename)
            n_reads_list.append(read_pattern[-1][0])  #TODO why is [0] needed to not have LNode([46])
            logging.info(f"Memory used in file_name loop after content file I/O: {get_mem_usage():.2f} GB")

        # Need maximum number of reads to initialize empty super dark array.
        self.max_reads = np.amax([*set(n_reads_list)])
        self.n_reads_list = n_reads_list

        # Sort file_list and n_reads_list by ascending number of reads to be used when creating super dark
        # and make file I/O more efficient/
        sorted_lists = sorted(zip(self.n_reads_list, self.file_list))
        self.n_reads_list, self.file_list = zip(*sorted_lists)

        timing_end_getmeta = time.time()
        elapsed_time = timing_end_getmeta - timing_start_getmeta
        print(f"Total time taken to get all files meta: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken to get all files meta: {elapsed_time:.2f} seconds")
        print(f"Memory used at the end of get_file_list_meta_rdmopen: {get_mem_usage():.2f} GB")
        logging.info(f"Memory used at the end of get_file_list_meta_rdmope: {get_mem_usage():.2f} GB")

    #def make_superdark_rdxrd_avg_rdmopen(self, max_reads=98, n_reads_list_sorted=None, file_list_sorted=None):
    def make_superdark_rdxrd_avg_rdmopen(self):
        """
        This method does a file I/O open, read, sum, and then average approach for every read
        available in each exposure/file used in creating the super dark cube. The method initializes an
        empty superdark cube with the dimensions in number of reads being the longest exposure in the
        filelist provided. Then starting with the first read on index 0, checks the file to be opened
        has a read frame matching the rd index. If it does not, representing a file with fewer reads than
        the maximum number of reads in the filelist, the file is not opened. If the file does contain a
        read frame for the read index rd, then it is opened, the data extracted and summed to create
        a running accumulation of signal; in each frame and a file counter is incremented. When finished
        for every file within the read index rd, the summed frame is divided by the file counter to give
        the mean signal.

        Parameters
        ----------

        """

        # self.n_reads_list = n_reads_list_sorted
        # self.file_list = file_list_sorted
        # self.max_reads = max_reads

        self.rdxrd_avg_cube = np.zeros((self.max_reads, 4096, 4096), dtype=np.float32)

        print("Testing read by read file I/O averaging.")
        print(f"Memory used at start of make_superdark_rdxrd_avg_rdmopen: {get_mem_usage():.2f} GB")
        timing_start_rxr_avg = time.time()
        i = 0

        logging.info("Testing read by read file I/O averaging.")
        logging.info(f"Memory used at start of make_superdark_rdxrd_avg_rdmopen: {get_mem_usage():.2f} GB")

        print(f"Memory used before read loop in rdxrd_avg_rdmopen : {get_mem_usage():.2f} GB")
        for rd in range(0, self.max_reads):
            summed_reads = np.zeros((4096, 4096), dtype='uint16')  # Empty 2D array to sum for each read from all files.
            logging.info(f"On read {rd} of {self.max_reads}")
            print(f"On read {rd} of {self.max_reads}")
            file_count_per_read = 0  # Counter for number of files with corresponding read index.
            for fn in range(0,len(self.file_list)):
                file_name = self.file_list[fn]
                file_path = self.input_path.joinpath(file_name)
                print(f"Memory used at start of file_name loop : {get_mem_usage():.2f} GB")
                print(F'on read {rd} and checking file {file_name} against number of reads '
                      F'in list {self.n_reads_list[fn]}')
                if rd <= self.n_reads_list[fn]:
                    # If the file to be opened has less than or equal to the number of maximum reads
                    # then open file and get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads/
                    try:
                        with rdm.open(file_path) as af:
                            logging.info(f"Opening file {file_path}")
                            print(f"Opening file {file_path}")
                            data = af.data
                            if isinstance(data, u.Quantity):  # Only access data from quantity object.
                                data = data.value
                                logging.info('Extracting data values from quantity object.')
                            summed_reads += data[rd, :, :]  # Get read according to rd index from data
                            file_count_per_read += 1  # Increase file counter
                    except Exception as e:
                        logging.error(f"Error opening file {file_path}: {e}")
                        print(f"Error opening file {file_path}: {e}")
                else:
                    # Skip this file if it has less reads than the read index.
                    print('skipping file', file_path)
                    continue

                print(f"Memory used at end of file_name loop before gc collect: {get_mem_usage():.2f} GB")
                del data
                gc.collect()
                print(f"Memory used at end of file_name loop after gc collect : {get_mem_usage():.2f} GB")

            print(file_count_per_read)
            averaged_read = summed_reads / file_count_per_read
            self.rdxrd_avg_cube[rd, :, :] = averaged_read
            print(f"Memory used at end of read index loop: {get_mem_usage():.2f} GB")

            #gc.collect()  # clean up memory
            #clipped_reads = sigma_clip(dark_read_cube, sigma_lower=sig_clip_md_low, sigma_upper=sig_clip_md_high,cenfunc=np.mean, axis=0, masked=False, copy=False)
            #self.super_dark[rd, :, :] = np.mean(clipped_reads, axis=0)
            #del clipped_reads
            #gc.collect()  # Clean up memory.

    def make_superdark_rdxrd_avg_asdfopen(self):
        """
        """

        # Display the directory name where the dark calibration files are located to make the master dark.
        logging.info(
            f"Using files from {os.path.dirname(self.input_data[0])} to construct super dark object."
        )

        # Find the dark calibration file with the most number of reads to initialize the super dark cube.
        tmp_reads = []
        for fl in range(0, len(self.input_data)):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_rds, _, _ = np.shape(tmp.tree["roman"]["data"])
            tmp_reads.append(n_rds)
            tmp.close()
        num_reads_set = [*set(tmp_reads)]
        del tmp_reads, tmp
        gc.collect()

        # The super dark length is the maximum number of reads in all dark calibration files to be used
        # when creating the dark reference file. Need to try over files with different lengths
        # to compute average read by read for all files
        self.super_dark = np.zeros(
            (np.max(num_reads_set), 4096, 4096), dtype=np.float32
        )
        # This method of opening and closing each file read by read is file I/O intensive however
        # it is efficient on memory usage.
        logging.info(
            "Reading dark asdf files read by read to compute average for master dark."
        )
        print("reading files")
        for rd in range(0, np.max(num_reads_set)):
            dark_read_cube = []
            logging.info(f"On read {rd} of {np.max(num_reads_set)}")
            print("read", rd)
            for fl in range(0, len(self.input_data)):
                print(fl, "file")
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                rd_tmp = tmp.tree["roman"]["data"]
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
        self.super_dark[:, :4, :] = 0.0
        self.super_dark[:, -4:, :] = 0.0
        self.super_dark[:, :, :4] = 0.0
        self.super_dark[:, :, -4:] = 0.0
        logging.info("Master dark attribute created.")


    def make_super_dark(self, raw_cube=None, sig_clip_md_low=3.0, sig_clip_md_high=3.0):
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

        # Display the directory name where the dark calibration files are located to make the master dark.
        logging.info(
            f"Using files from {os.path.dirname(self.input_data[0])} to construct super dark object."
        )

        # Find the dark calibration file with the most number of reads to initialize the super dark cube.
        tmp_reads = []
        for fl in range(0, len(self.input_data)):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_rds, _, _ = np.shape(tmp.tree["roman"]["data"])
            tmp_reads.append(n_rds)
            tmp.close()
        num_reads_set = [*set(tmp_reads)]
        del tmp_reads, tmp
        gc.collect()

        # The super dark length is the maximum number of reads in all dark calibration files to be used
        # when creating the dark reference file. Need to try over files with different lengths
        # to compute average read by read for all files
        self.super_dark = np.zeros(
            (np.max(num_reads_set), 4096, 4096), dtype=np.float32
        )
        # This method of opening and closing each file read by read is file I/O intensive however
        # it is efficient on memory usage.
        logging.info(
            "Reading dark asdf files read by read to compute average for master dark."
        )
        print("reading files")
        for rd in range(0, np.max(num_reads_set)):
            dark_read_cube = []
            logging.info(f"On read {rd} of {np.max(num_reads_set)}")
            print("read", rd)
            for fl in range(0, len(self.input_data)):
                print(fl, "file")
                tmp = asdf.open(self.input_data[fl], validate_on_read=False)
                rd_tmp = tmp.tree["roman"]["data"]
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
        self.super_dark[:, :4, :] = 0.0
        self.super_dark[:, -4:, :] = 0.0
        self.super_dark[:, :, :4] = 0.0
        self.super_dark[:, :, -4:] = 0.0
        logging.info("Master dark attribute created.")

    def save_suoer_dark(self, superdark_outfile=None):
        """
        The method save_super_dark with default conditions will write the super dark cube into an asdf
        file for each detector in the directory from which the input files where pointed to and used to
        construct the super dark read cube. A user can specify an absolute path or relative file string
        to write the super dark file name to disk.

        Parameters
        ----------
        superdark_outfile: str; default = None
            File string. Absolute or relative path for optional input.
            By default, None is provided but the method below generates the asdf file string from meta
            data of the input files such as date and detector number  (i.e. WFI01) in the filename.
        """

        meta_superdark = {'pedigree': "GROUND", 'description': "Super dark internal reference file calibration product"
                                                               "generated from Reference File Pipeline.",
                          'date': Time(datetime.datetime.now()), 'detector': self.meta['instrument']['detector']}
        if superdark_outfile is None:
            superdark_outfile = Path(self.input_data[0] + '/' + meta_superdark['detector'] + '_super_dark.asdf')
        else:
            superdark_outfile = 'roman_super_dark.asdf'
        self.check_output_file(superdark_outfile)
        logging.info('Saving super dark to disk.')

        af = asdf.AsdfFile()
        af.tree = {'meta': meta_superdark, 'data': self.super_dark}
        af.write_to(superdark_outfile)
