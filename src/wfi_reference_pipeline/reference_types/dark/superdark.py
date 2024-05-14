import datetime
import gc
import logging
import time
import asdf
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy import units as u
from pathlib import Path
import roman_datamodels as rdm
import psutil
from datetime import datetime


def get_mem_usage():
    """
    Function to return memory usage throughout module.

    Returns
    ----------
    memory_usage; float
        Memory in Gigabytes being used.

    """
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
        self.superdark_A = None
        self.superdark_B = None
        self.superdark_C = None
        self.meta_data = None
        self.max_reads = None
        self.n_reads_list = None
        self.clipped_reads = None
        self.reads_from_all_files = None

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

    def make_superdark_method_A(self,
                                sig_clip_sd_low=3.0,
                                sig_clip_sd_high=3.0,
                                max_reads=98,
                                n_reads_list_sorted=None,
                                file_list_sorted=None):

        """
        This method does a file I/O open, read, append to a temporary read cube, sigma clip, and then average
        approach for every read available in each exposure/file used in creating the super dark cube. Starting
        with the first read index 0, checks the file to be opened has a read frame matching the rd index. If it
        does not, representing a file with fewer reads than the maximum number of reads from the file list,
        the file is not opened. If the file does contain a read frame for the read index rd, then it is opened,
        the data extracted for that read only and appended to a temporary cube representing that read for all
        of the files in the file list. That cube is then sigma clipped to remove outliers - aka cosmic rays -
        and averaged to produce mean pixel dark value for that read up the ramp.

        Parameters
        ----------
        sig_clip_sd_low: float; default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high: float; default = 3.0
            Upper bound limit to filter data
        max_reads: int; default = 98
            The number of reads in the long dark exposures from the in-flight calibration plan.
        n_reads_list_sorted: list, default = None
            A list of integers representing the ordered number of reads of each dark exposure from
            the in-flight calibration plan. Short darks have 46 reads. Long darks have 98 reads.
        file_list_sorted: list, default = None
            A list of the sorted by filenames and shortest to longest number of reads in each
            in-flight calibration plan dark exposure.

        """

        self.n_reads_list = n_reads_list_sorted
        self.file_list = file_list_sorted
        self.max_reads = max_reads
        self.superdark_A = np.zeros((self.max_reads, 4096, 4096), dtype=np.float32)

        timing_start_method_A = time.time()
        print("Testing super dark method A.")
        print(f"Memory used at start of method A: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method A.")
        logging.info(f"Memory used at start of method A: {get_mem_usage():.2f} GB")
        for rd in range(0, self.max_reads, 50):
            logging.info(f"On read {rd} of {self.max_reads}")
            print(f"On read {rd} of {self.max_reads}")
            reads_from_all_files = []
            for fn in range(0, len(self.file_list)):
                file_name = self.file_list[fn]
                file_path = self.input_path.joinpath(file_name)
                if rd <= self.n_reads_list[fn]:
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    try:
                        #TODO figure out why appending a frame is doubling memory every file
                        # This is different from testing a year ago
                        with rdm.open(file_path) as af:
                            logging.info(f"Opening file {file_path}")
                            #print(f"Opening file {file_path}")
                            tmp = af.data.value
                            print(tmp)
                            # if isinstance(data, u.Quantity):  # Only access data from quantity object.
                            #     reads_from_all_files.append(data[rd, :, :].value)
                            # else:
                        reads_from_all_files.append(tmp[fn, :, :])

                            #print(data[rd, :, :], np.shape(data[rd, :, :]))

                        print(np.shape(reads_from_all_files))
                        del af, tmp
                        gc.collect()
                        print(f"Memory in file loop method A: {get_mem_usage():.2f} GB")
                        #print(reads_from_all_files)

                        # with asdf.open(file_path) as tmp:
                        #     tmp_rd = tmp.tree["roman"]["data"].value
                        # reads_from_all_files.append(tmp_rd[fn, :, :])
                        # del tmp, data
                        # gc.collect()


                    except Exception as e:
                        logging.error(f"Error opening file {file_path}: {e}")
                        print(f"Error opening file {file_path}: {e}")
                else:
                    # Skip this file if it has less reads than the read index.
                    # print('skipping file', file_path)
                    continue

                print(f"Memory at end of file loop in method A: {get_mem_usage():.2f} GB")

            break

            clipped_reads = sigma_clip(reads_from_all_files,
                                       sigma_lower=sig_clip_sd_low,
                                       sigma_upper=sig_clip_sd_high,
                                       cenfunc=np.mean,
                                       axis=0,
                                       masked=False,
                                       copy=False)
            self.superdark_A[rd, :, :] = np.mean(clipped_reads, axis=0)
            print(f"Memory used at end of read index loop method A: {get_mem_usage():.2f} GB")
            del clipped_reads
            gc.collect()

        timing_end_method_A = time.time()
        elapsed_time = timing_end_method_A - timing_start_method_A
        print(f"Total time taken for method A: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method A: {elapsed_time:.2f} seconds")

    def make_superdark_method_B(self,
                                max_reads=98,
                                n_reads_list_sorted=None,
                                file_list_sorted=None):
        """
        This method does a file I/O open, read, sum, and then average approach for every read
        available in each exposure/file used in creating the super dark cube. The method initializes an
        empty super dark cube with a length corresponding to the number of reads in the longest exposure in the
        file list provided. Then starting with the first read index 0, checks the file to be opened
        has a read frame matching the rd index. If it does not, representing a file with fewer reads than
        the maximum number of reads from the file list, the file is not opened. If the file does contain a
        read frame for the read index rd, then it is opened, the data extracted for that read only and summed to create
        a running accumulation of signal in each frame and a file counter is incremented. When finished
        for every file within the read index rd, the summed frame is divided by the file counter to give
        the mean signal.

        This does not allow for sigma clipping or removing cosmic rays which is a concern but this is probab;y
        the least expensive memory utilized approach.

        Parameters
        ----------
        max_reads: int; default = 98
            The number of reads in the long dark exposures from the in-flight calibration plan.
        n_reads_list_sorted: list, default = None
            A list of integers representing the ordered number of reads of each dark exposure from
            the in-flight calibration plan. Short darks have 46 reads. Long darks have 98 reads.
        file_list_sorted: list, default = None
            A list of the sorted by filenames and shortest to longest number of reads in each
            in-flight calibration plan dark exposure.
        """

        self.n_reads_list = n_reads_list_sorted
        self.file_list = file_list_sorted
        self.max_reads = max_reads

        self.superdark_B = np.zeros((self.max_reads, 4096, 4096), dtype=np.float32)

        timing_start_method_B = time.time()
        print("Testing super dark method B.")
        print(f"Memory used at start of method B: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method B.")
        logging.info(f"Memory used at start of method B: {get_mem_usage():.2f} GB")
        for rd in range(0, self.max_reads, 40):
            summed_reads_from_all_files = np.zeros((4096, 4096), dtype='uint16')  # Empty 2D array to sum for each read from all files.
            logging.info(f"On read {rd} of {self.max_reads}")
            print(f"On read {rd} of {self.max_reads}")
            file_count_per_read = 0  # Counter for number of files with corresponding read index.
            for fn in range(0,len(self.file_list)):
                file_name = self.file_list[fn]
                file_path = self.input_path.joinpath(file_name)
                if rd <= self.n_reads_list[fn]:
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    try:
                        with rdm.open(file_path) as af:
                            logging.info(f"Opening file {file_path}")
                            #print(f"Opening file {file_path}")
                            data = af.data
                            if isinstance(data, u.Quantity):  # Only access data from quantity object.
                                data = data.value
                                logging.info('Extracting data values from quantity object.')
                            summed_reads_from_all_files += data[rd, :, :]  # Get read according to rd index from data
                            file_count_per_read += 1  # Increase file counter
                            print(f"Memory in file loop method B: {get_mem_usage():.2f} GB")
                    except Exception as e:
                        logging.error(f"Error opening file {file_path}: {e}")
                        print(f"Error opening file {file_path}: {e}")
                else:
                    # Skip this file if it has less reads than the read index.
                    # print('skipping file', file_path)
                    continue
                print(f"Memory at end of file loop in method B: {get_mem_usage():.2f} GB")
                del data
                gc.collect()

            averaged_read = summed_reads_from_all_files / file_count_per_read
            self.superdark_B[rd, :, :] = averaged_read
            print(f"Memory used at end of read index loop method B: {get_mem_usage():.2f} GB")

        timing_end_method_B = time.time()
        elapsed_time = timing_end_method_B - timing_start_method_B
        print(f"Total time taken for method B: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method B: {elapsed_time:.2f} seconds")

    def make_superdark_method_C(self,
                                sig_clip_sd_low=3.0,
                                sig_clip_sd_high=3.0,
                                max_reads=98,
                                n_reads_list_sorted=None,
                                file_list_sorted=None):

        """
        This method does a file I/O open, read, append to a temporary read cube, sigma clip, and then average
        approach for every read available in each exposure/file used in creating the super dark cube. Starting
        with the first read index 0, checks the file to be opened has a read frame matching the rd index. If it
        does not, representing a file with fewer reads than the maximum number of reads from the file list,
        the file is not opened. If the file does contain a read frame for the read index rd, then it is opened,
        the data extracted for that read only and appended to a temporary cube representing that read for all
        of the files in the file list. That cube is then sigma clipped to remove outliers - aka cosmic rays -
        and averaged to produce mean pixel dark value for that read up the ramp.

        Parameters
        ----------
        sig_clip_sd_low: float; default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high: float; default = 3.0
            Upper bound limit to filter data
        max_reads: int; default = 98
            The number of reads in the long dark exposures from the in-flight calibration plan.
        n_reads_list_sorted: list, default = None
            A list of integers representing the ordered number of reads of each dark exposure from
            the in-flight calibration plan. Short darks have 46 reads. Long darks have 98 reads.
        file_list_sorted: list, default = None
            A list of the sorted by filenames and shortest to longest number of reads in each
            in-flight calibration plan dark exposure.

        # Method C last output. Need to get logging working
        # Memory in file loop method C: 11.21 GB
        # Memory at end of file loop in method C: 11.21 GB
        # Sigma clipping reads from all files for read
        # Memory used at end of read index loop method C: 12.61 GB
        # Read loop C time: 339.90 seconds
        # Current date and time: 2024-05-08 11:33:18.754655
        # Total time taken for method C: 41745.84 seconds

        """
        current_datetime = datetime.now()
        print("Current date and time:", current_datetime)

        # Uncomment these or figure out a way to have inputs
        # self.n_reads_list = n_reads_list_sorted
        # self.file_list = file_list_sorted
        # self.max_reads = max_reads
        self.superdark_C = np.zeros((self.max_reads, 4096, 4096), dtype=np.float32)

        timing_start_method_C = time.time()
        print("Testing super dark method C.")
        print(f"Memory used at start of method C: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method C.")
        logging.info(f"Memory used at start of method C: {get_mem_usage():.2f} GB")
        for rd in range(0, self.max_reads):
            timing_start_method_c_rd_loop = time.time()
            logging.info(f"On read {rd} of {self.max_reads}")
            print(f"On read {rd} of {self.max_reads}")
            self.reads_from_all_files = np.zeros((len(self.file_list), 4096, 4096), dtype=np.float32)
            for fn in range(0, len(self.file_list)):
                file_name = self.file_list[fn]
                file_path = self.input_path.joinpath(file_name)
                if rd <= self.n_reads_list[fn]:
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    try:
                        with rdm.open(file_path) as af:
                            logging.info(f"Opening file {file_path}")
                            #print(f"Opening file {file_path}")
                            data = af.data
                            if isinstance(data, u.Quantity):  # Only access data from quantity object.
                                data = data.value
                            self.reads_from_all_files[fn, :, :] = data[rd, :, :]
                            print(f"Memory in file loop method C: {get_mem_usage():.2f} GB")
                    except Exception as e:
                        logging.error(f"Error opening file {file_path}: {e}")
                        print(f"Error opening file {file_path}: {e}")
                else:
                    # Skip this file if it has less reads than the read index.
                    print('skipping file', file_path)
                    continue

                print(f"Memory at end of file loop in method C: {get_mem_usage():.2f} GB")
                del af, data
                gc.collect()

            nonzero_slices = np.where(~np.all(self.reads_from_all_files == 0, axis=(1, 2)))[0]
            reduced_array = self.reads_from_all_files[nonzero_slices]
            # Remove NaNs from the array using boolean indexing
            reduced_array_no_nans = reduced_array[~np.isnan(reduced_array)]
            print('Sigma clipping reads from all files for read')
            clipped_reads = sigma_clip(reduced_array_no_nans,
                                        sigma_lower=sig_clip_sd_low,
                                        sigma_upper=sig_clip_sd_high,
                                        cenfunc=np.mean,
                                        axis=0,
                                        masked=False,
                                        copy=False)
            self.superdark_C[rd, :, :] = np.mean(clipped_reads, axis=0)

            print(f"Memory used at end of read index loop method C: {get_mem_usage():.2f} GB")
            del reduced_array, clipped_reads, self.reads_from_all_files
            gc.collect()
            timing_end_method_c_rd_loop = time.time()
            elapsed_time = timing_end_method_c_rd_loop - timing_start_method_c_rd_loop
            print(f"Read loop C time: {elapsed_time:.2f} seconds")
            #break

            current_datetime = datetime.now()
            print("Current date and time:", current_datetime)

        timing_end_method_C = time.time()
        elapsed_time = timing_end_method_C - timing_start_method_C
        print(f"Total time taken for method C: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method CA: {elapsed_time:.2f} seconds")

        self.superdark = self.superdark_C

    def write_superdark(self, outfile=None):
        """
        The method save_super_dark with default conditions will write the super dark cube into an asdf
        file for each detector in the directory from which the input files where pointed to and used to
        construct the super dark read cube. A user can specify an absolute path or relative file string
        to write the super dark file name to disk.

        Parameters
        ----------
        outfile: str; default = None
            File string. Absolute or relative path for optional input.
            By default, None is provided but the method below generates the asdf file string from meta
            data of the input files such as date and detector number  (i.e. WFI01) in the filename.
        """

        # set reference pixel border to zero for super dark
        # this needs to be done differently for multi sub array jigsaw handling
        # move to when making the mask and final stitching together different pieces to do the border
        self.superdark[:, :4, :] = 0.0
        self.superdark[:, -4:, :] = 0.0
        self.superdark[:, :, :4] = 0.0
        self.superdark[:, :, -4:] = 0.0

        meta_superdark = {'pedigree': "DUMMY",
                          'description': "Super dark file calibration product "
                                         "generated from Reference File Pipeline.",
                          'date': Time(datetime.now()),
                          'detector': 'WFO01',
                          'filelist': self.file_list}

        #TODO need filename to have date in YYYYMMDD format probably....need to get meta data from
        # files to populate superdark meta - what is relevant besides detector and filelist and mode?

        if outfile is None:
            outfile = Path(self.input_path) / (meta_superdark['detector'] + '_superdark.asdf')
        #self.check_outfile(superdark_outfile)
        logging.info('Saving superdark asdf to disk.')

        af = asdf.AsdfFile()
        af.tree = {'meta': meta_superdark,
                   'data': self.superdark}
        af.write_to(outfile)
