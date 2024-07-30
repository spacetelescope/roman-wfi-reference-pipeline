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
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os


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


def process_file(file_path, read_i):
    """
    Helper function to process a single file.

    Parameters
    ----------
    file_path : Path
        Path to the file to be processed.
    read_i : int
        Read index.

    Returns
    ----------
    np.ndarray
        Array of data for the given read index.
    """

    try:
        with asdf.open(file_path) as af:
            logging.info(f"Opening file {file_path}")
            if isinstance(af.tree['roman']['data'], u.Quantity):  # Only access data from quantity object.
                return af.tree['roman']['data'][read_i, :, :].value
            else:
                return af.tree['roman']['data'][read_i, :, :]
    except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
        logging.warning(f"Could not open {file_path} - {e}")


def process_files_in_batches(file_path, file_list, batch_size, read_i):
    """
    Processes a list of files in batches to read data for a specific read index. This function divides
    the list of files into batches, processes each batch in parallel using a ThreadPoolExecutor,
    and reads data from each file for the given read index. Results from all files are aggregated
    into a single list.

    Parameters
    ----------
    file_path : Path
        Path to the files to be processed.
    file_list : list of str
        List of file names to be processed at the file_path.
    batch_size : int
        Number of files to process in parallel at a time.
    read_i : int
        The index of the read or data slice to extract from each file.

    Returns
    -------
    list of np.ndarray
        List of numpy arrays containing data for the specified read index from each file.
    """

    all_results = []
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        # Specify that the batch size is the max number of workers or cores to open files.
        # Limit one core per file.
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process_file, file_path.joinpath(file), read_i) for file in batch]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)
    return all_results


class SuperDark:
    """
    SuperDark() is a class that will ingest raw L1 dark calibration files and average every read for
    as many exposures as there are available for that read to create a superdark.asdf file. This file
    is the assumed input into the Dark() module in the RFP to create resampled dark calibration
    reference files for a specific MA Table.
    """

    def __init__(
        self,
        input_path,
        file_list=None,
        n_reads_list=None,
        short_dark_file_list=None,
        long_dark_file_list=None,
        outfile="roman_superdark.asdf",
    ):
        """
        Parameters
        ----------
        input_path: str,
            Path to input directory where files are located.
        file_list: list,
            List of files in the input_directory
        short_dark_file_list : list, default = None
            List of short dark exposure files.
        long_dark_file_list : list, default = None
            List of long dark exposure files.
        outfile: str, default="roman_superdark.asdf"
            File name written to disk.
        """

        # Specify file lists.
        if file_list is None and (short_dark_file_list is None or long_dark_file_list is None):
            raise ValueError("Either 'file_list' must be provided, or both "
                             "'short_dark_file_list' and 'long_dark_file_list' must be provided.")

        self.input_path = Path(input_path)
        self.file_list = None
        self.n_reads_list = n_reads_list

        # Initialize with file_list.
        if file_list:
            self.file_list = sorted(file_list)
            if n_reads_list:
                self.max_reads = np.amax(n_reads_list)
        # Initialize with short_dark_file_list and long_dark_file_list
        elif short_dark_file_list and long_dark_file_list:
            self.short_dark_file_list = sorted(short_dark_file_list)
            self.short_dark_num_reads = 46
            self.long_dark_file_list = sorted(long_dark_file_list)
            self.long_dark_num_reads = 98
            self.file_list = sorted(short_dark_file_list + long_dark_file_list)
        else:
            raise ValueError(
                "Invalid input combination: both 'short_dark_file_list' and "
                "'long_dark_file_list' must be provided together.")

        # Get WFIXX string
        wfixx_strings = [re.search(r'(WFI\d{2})', file).group(1) for file in self.file_list if
                         re.search(r'(WFI\d{2})', file)]
        self.wfixx_string = list(set(wfixx_strings))  # Remove duplicates if needed
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = str(self.input_path / (self.wfixx_string[0] + '_superdark.asdf'))

        # The attribute that contains the i'th read from all files or exposures. This is the array
        # that is sigma clipped or filtered to remove hot and dead pixels and cosmic rays.
        self.read_i_from_all_files = None
        # The array of filtered reads from all files for the i'th read of the superdark.
        self.clipped_reads = None

        self.superdark = None
        self.superdark_a = None
        self.superdark_b = None
        self.superdark_c = None
        self.superdark_d = None
        self.superdark_e = None
        self.superdark_f = None
        self.superdark_g = None

        # Meta data for RFP tracking and usage. Not a CRDS delivered product.
        self.meta_data = {'pedigree': "DUMMY",
                          'description': "Super dark file calibration product "
                                         "generated from Reference File Pipeline.",
                          'date': Time(datetime.now()),
                          'detector': self.wfixx_string,
                          'filelist': self.file_list}

    def make_superdark_method_c(self,
                                sig_clip_sd_low=3.0,
                                sig_clip_sd_high=3.0,
                                open_type='asdf'):
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
        open_type: string, default = 'rdm'
            Default string for file i/o is romandatamodels or rdm. Also allowed is 'asdf'
        """
        current_datetime = datetime.now()
        print("Current date and time:", current_datetime)

        # Uncomment these or figure out a way to have inputs
        if self.n_reads_list is None:
            raise ValueError("The variable n_reads_list cannot be None for method c.")
        self.superdark_c = np.zeros((self.max_reads, 4096, 4096), dtype=np.float32)

        # Check if open_type is a string and matches 'asdf' or 'rdm'
        if isinstance(open_type, str) and open_type in ['asdf', 'rdm']:
            logging.info(f"Valid open_type: {open_type}")
        else:
            raise ValueError("Invalid open_type. Must be 'asdf' or 'rdm'.")

        timing_start_method_c = time.time()
        print("Testing super dark method c.")
        print(f"Memory used at start of method c: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method c.")
        logging.info(f"Memory used at start of method c: {get_mem_usage():.2f} GB")
        for read_i in range(0, self.max_reads):
            timing_start_method_c_rd_loop = time.time()
            logging.info(f"On read {read_i} of {self.max_reads}")
            print(f"On read {read_i} of {self.max_reads}")
            self.read_i_from_all_files = np.zeros((len(self.file_list), 4096, 4096), dtype=np.float32)
            for file_f in range(0, len(self.file_list)):
                file_name = self.file_list[file_f]
                file_path = self.input_path.joinpath(file_name)
                if read_i < self.n_reads_list[file_f]:
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    if open_type == 'rdm':
                        try:
                            with rdm.open(file_path) as af:
                                logging.info(f"Opening file {file_path}")
                                if isinstance(af.data, u.Quantity):  # Only access data from quantity object.
                                    self.read_i_from_all_files[file_f, :, :] = af.data.value[read_i, :, :]
                                else:
                                    self.read_i_from_all_files[file_f, :, :] = af.data[read_i, :, :]
                            print(f"Memory after file rdm open method c: {get_mem_usage():.2f} GB")
                        except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
                            logging.warning(f"Could not open {file_path} - {e}")
                    if open_type == 'asdf':
                        try:
                            with asdf.open(file_path) as af:
                                logging.info(f"Opening file {file_path}")
                                if isinstance(af.tree['roman']['data'], u.Quantity):  # Only access data from quantity object.
                                    self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                                else:
                                    self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                            print(f"Memory after file asdf open method c: {get_mem_usage():.2f} GB")
                        except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
                            logging.warning(f"Could not open {file_path} - {e}")
                else:
                    # Skip this file if it has less reads than the read index.
                    print('skipping file', file_path)
                    continue
                print(f"Memory at end of file loop in method c: {get_mem_usage():.2f} GB")
                del af
                gc.collect()

            # Remove any frames or reads that were skipped by removing all 2D arrays of all zeros.
            nonzero_reads_indices = np.where(~np.all(self.read_i_from_all_files == 0, axis=(1, 2)))[0]
            #print(nonzero_reads_indices)
            read_i_reduced_cube = self.read_i_from_all_files[nonzero_reads_indices]
            # Remove NaNs from raw data from the cube using boolean indexing.
            #self.read_i_reduced_cube_no_nans = self.read_i_reduced_cube[~np.isnan(self.read_i_reduced_cube)]
            print(f'Sigma clipping reads from all files for read_i: {read_i}')
            # self.clipped_reads = sigma_clip(self.read_i_reduced_cube_no_nans,
            #                            sigma_lower=sig_clip_sd_low,
            #                            sigma_upper=sig_clip_sd_high,
            #                            cenfunc=np.mean,
            #                            axis=0,
            #                            masked=False,
            #                            copy=False)
            clipped_reads = sigma_clip(read_i_reduced_cube,
                                       sigma_lower=sig_clip_sd_low,
                                       sigma_upper=sig_clip_sd_high,
                                       cenfunc=np.mean,
                                       axis=0,
                                       masked=False,
                                       copy=False)
            self.superdark_c[read_i, :, :] = np.mean(clipped_reads, axis=0)

            print(f"Memory used at end of read index loop method c: {get_mem_usage():.2f} GB")
            #del self.read_i_from_all_files, read_i_reduced_cube, read_i_reduced_cube_no_nans, clipped_reads,
            gc.collect()
            timing_end_method_c_rd_loop = time.time()
            elapsed_time = timing_end_method_c_rd_loop - timing_start_method_c_rd_loop
            print(f"Read loop c time: {elapsed_time:.2f} seconds")
            # break

            current_datetime = datetime.now()
            print("Current date and time:", current_datetime)

        timing_end_method_c = time.time()
        elapsed_time = timing_end_method_c - timing_start_method_c
        print(f"Total time taken for method c: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method c: {elapsed_time:.2f} seconds")

        self.superdark = self.superdark_c

    def make_superdark_method_d(self,
                                sig_clip_sd_low=3.0,
                                sig_clip_sd_high=3.0):
        """
        This method does a file I/O open, read, and append to a temporary cube, sigma clip, and then average
        approach for every read in both short and long darks in creating the super dark cube. Starting with
        read index 0 all files that have the read index in the allowed range will be opened and the frame from
        each exposure extracted and inserted into a temporary cube representing the number of files available
        for that read.

        Parameters
        ----------
        sig_clip_sd_low : float, default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high : float, default = 3.0
            Upper bound limit to filter data.
        """
        current_datetime = datetime.now()
        print("Current date and time:", current_datetime)

        timing_start_method_d = time.time()
        print("Testing super dark method d.")
        print(f"Memory used at start of method d: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method d.")
        logging.info(f"Memory used at start of method d: {get_mem_usage():.2f} GB")

        self.superdark_d = np.zeros((self.long_dark_num_reads, 4096, 4096), dtype=np.float32)

        for read_i in range(0, self.long_dark_num_reads):
            timing_start_method_d_rd_loop = time.time()
            logging.info(f"On read {read_i} of {self.long_dark_num_reads}")
            print(f"On read {read_i} of {self.long_dark_num_reads}")

            self.read_i_from_all_files = np.zeros((len(self.short_dark_file_list) +
                                                   len(self.long_dark_file_list),
                                                   4096, 4096), dtype=np.float32)

            for file_f in range(max(len(self.short_dark_file_list), len(self.long_dark_file_list))):
                # Process short dark files.
                if file_f < len(self.short_dark_file_list) and read_i < self.short_dark_num_reads:
                    short_dark_file_name = self.short_dark_file_list[file_f]
                    short_dark_file_path = self.input_path.joinpath(short_dark_file_name)
                    try:
                        with asdf.open(short_dark_file_path) as af:
                            logging.info(f"Opening file {short_dark_file_path}")
                            if isinstance(af.tree['roman']['data'], u.Quantity):  # Only access data from quantity object.
                                self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                            else:
                                self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                        print(f"Memory after file asdf open method d: {get_mem_usage():.2f} GB")
                    except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
                        logging.warning(f"Could not open {short_dark_file_path} - {e}")
                        continue
                # Process long dark files.
                if file_f < len(self.long_dark_file_list):
                    long_dark_file_name = self.long_dark_file_list[file_f]
                    long_dark_file_path = self.input_path.joinpath(long_dark_file_name)
                    try:
                        with asdf.open(long_dark_file_path) as af:
                            logging.info(f"Opening file {long_dark_file_path}")
                            if isinstance(af.tree['roman']['data'], u.Quantity):  # Only access data from quantity object.
                                self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                            else:
                                self.read_i_from_all_files[file_f, :, :] = af.tree['roman']['data'][read_i, :, :]
                        print(f"Memory after file asdf open method d: {get_mem_usage():.2f} GB")
                    except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
                        logging.warning(f"Could not open {long_dark_file_path} - {e}")
                print(f"Memory at end of file loop in method d: {get_mem_usage():.2f} GB")

            print(f'Sigma clipping reads from all files for read_i: {read_i}')
            clipped_reads = sigma_clip(self.read_i_from_all_files,
                                       sigma_lower=sig_clip_sd_low,
                                       sigma_upper=sig_clip_sd_high,
                                       cenfunc=np.mean,
                                       axis=0,
                                       masked=False,
                                       copy=False)
            self.superdark_d[read_i, :, :] = np.mean(clipped_reads, axis=0)

            print(f"Memory used at end of read index loop method d: {get_mem_usage():.2f} GB")
            del clipped_reads, self.read_i_from_all_files
            gc.collect()
            timing_end_method_d_rd_loop = time.time()
            elapsed_time = timing_end_method_d_rd_loop - timing_start_method_d_rd_loop
            print(f"Read loop d time: {elapsed_time:.2f} seconds")

            current_datetime = datetime.now()
            print("Current date and time:", current_datetime)

        timing_end_method_d = time.time()
        elapsed_time = timing_end_method_d - timing_start_method_d
        print(f"Total time taken for method d: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method d: {elapsed_time:.2f} seconds")

        self.superdark = self.superdark_d

    def make_superdark_method_e(self,
                                sig_clip_sd_low=3.0,
                                sig_clip_sd_high=3.0,
                                short_batch_size=4,
                                long_batch_size=4):
        """
        This method does a file I/O open, read, and append to a temporary cube, sigma clip, and then average
        approach for every read in both short and long darks in creating the super dark cube. Starting with
        read index 0 all files that have the read index in the allowed range will be opened and the frame from
        each exposure extracted and inserted into a temporary cube representing the number of files available
        for that read.

        Parameters
        ----------
        sig_clip_sd_low : float, default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high : float, default = 3.0
            Upper bound limit to filter data.
        short_batch_size : int, default = 4
            Number of short dark files to process in parallel at a time.
        long_batch_size : int, default = 4
            Number of long dark files to process in parallel at a time.
        """
        current_datetime = datetime.now()
        print("Current date and time:", current_datetime)
        timing_start_method_e = time.time()
        print(f"Testing super dark method e with short batch size of {short_batch_size} and long batch"
              f" size of {long_batch_size}.")
        print(f"Memory used at start of method e: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method e.")
        logging.info(f"Memory used at start of method e: {get_mem_usage():.2f} GB")

        self.superdark_e = np.zeros((self.long_dark_num_reads, 4096, 4096), dtype=np.float32)
        # Loop over read to construct superdark of length of long dark reads.
        # Going into each file for every i'th read or read_i index.
        for read_i in range(0, self.long_dark_num_reads):
            timing_start_method_e_file_loop = time.time()
            logging.info(f"On read {read_i} of {self.long_dark_num_reads}")
            print(f"On read {read_i} of {self.long_dark_num_reads}")

            # Determine the number of files to process for the current read index.
            if read_i < self.short_dark_num_reads:
                num_files = len(self.short_dark_file_list) + len(self.long_dark_file_list)
            else:
                num_files = len(self.long_dark_file_list)
            # Create temporary array for i'th read from all files.
            self.read_i_from_all_files = np.zeros((num_files, 4096, 4096), dtype=np.float32)

            short_dark_results = []
            # Process short dark files in batches if the read index is within the range of short dark reads
            if read_i < self.short_dark_num_reads:
                short_dark_results = process_files_in_batches(self.input_path,
                                                              self.short_dark_file_list,
                                                              short_batch_size,
                                                              read_i)
                for i, result in enumerate(short_dark_results):
                    if result is not None:
                        logging.info(f"Assigning result from short dark file to index {i} in supderdark "
                                     f"for read {read_i}")
                        self.read_i_from_all_files[i, :, :] = result

            # Need start at the short dark results to ensure correct placement and not overwrite short dark results
            # when doing long dark parallel processing.
            long_dark_results = process_files_in_batches(self.input_path,
                                                         self.long_dark_file_list,
                                                         long_batch_size,
                                                         read_i)
            for i, result in enumerate(long_dark_results, start=len(short_dark_results)):
                if result is not None:
                    logging.info(f"Assigning result from long dark file to index {i} in superdark"
                                 f"for read {read_i}")
                    self.read_i_from_all_files[i, :, :] = result

            timing_end_method_e_file_loop = time.time()
            elapsed_file_loop_time = timing_end_method_e_file_loop - timing_start_method_e_file_loop
            print(f"File loop time: {elapsed_file_loop_time:.2f} seconds")

            timing_start_sigmaclipmean = time.time()

            if np.isnan(self.read_i_from_all_files[i, :, :]).any():
                print('NaNs found in read_i_from_all_files')
            print(f'Sigma clipping reads from all files for read_i: {read_i}')
            clipped_reads = sigma_clip(self.read_i_from_all_files,
                                       sigma_lower=sig_clip_sd_low,
                                       sigma_upper=sig_clip_sd_high,
                                       cenfunc=np.mean,
                                       axis=0,
                                       masked=False,
                                       copy=False)
            if np.isnan(clipped_reads).any():
                print('NaNs found in clipped reads')
            self.superdark_e[read_i, :, :] = np.mean(clipped_reads, axis=0)

            print(f"Memory used at end of read index loop method e: {get_mem_usage():.2f} GB")
            del clipped_reads, self.read_i_from_all_files
            gc.collect()
            timing_end_sigmaclipmean = time.time()
            time_sigmaclipmean = timing_end_sigmaclipmean - timing_start_sigmaclipmean
            print(f"Sigma clip and average time: {time_sigmaclipmean:.2f} seconds")
            time_fileloop_and_sigmaclipmean = timing_end_sigmaclipmean - timing_start_method_e_file_loop
            print(f"File loop and sigma clip and average time: {time_fileloop_and_sigmaclipmean:.2f} seconds")

        timing_end_method_e = time.time()
        elapsed_time = timing_end_method_e - timing_start_method_e
        print(f"Total time taken for method e: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method e: {elapsed_time:.2f} seconds")
        self.superdark = self.superdark_e

    def make_superdark_method_f(self, sig_clip_sd_low=3.0, sig_clip_sd_high=3.0):
        """
        Create a super dark cube by processing reads from multiple files and sigma clipping outliers.

        Parameters
        ----------
        sig_clip_sd_low : float, default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high : float, default = 3.0
            Upper bound limit to filter data.
        """
        current_datetime = datetime.now()
        print("Current date and time:", current_datetime)
        timing_start_method_f = time.time()
        print(f"Testing super dark method f.")
        print(f"Memory used at start of method f: {get_mem_usage():.2f} GB")
        logging.info("Testing super dark method f.")
        logging.info(f"Memory used at start of method f: {get_mem_usage():.2f} GB")

        self.superdark_f = np.zeros((self.long_dark_num_reads, 4096, 4096), dtype=np.float32)

        #TODO this is actively in development and under construction by Rick Cosentino

        for read_i in range(0, self.long_dark_num_reads):
            timing_start_method_f_rd_loop = time.time()
            logging.info(f"On read {read_i} of {self.long_dark_num_reads}")
            print(f"On read {read_i} of {self.long_dark_num_reads}")

            # TODO This has not been fully tested and debugged yet.....
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_quadrant, read_i, self.file_list, self.n_reads_list, self.input_path,
                                    quad, sig_clip_sd_low, sig_clip_sd_high): quad
                    for quad in ['quad1', 'quad2', 'quad3', 'quad4']
                }

                results = {}
                for future in futures:
                    quad = futures[future]
                    results[quad] = future.result()

            self.superdark_f[read_i, :, :] = self.assemble_quadrants(
                results['quad1'], results['quad2'], results['quad3'], results['quad4']
            )

            print(f"Memory used at end of read index loop method f: {get_mem_usage():.2f} GB")
            gc.collect()
            timing_end_method_f_rd_loop = time.time()
            elapsed_time = timing_end_method_f_rd_loop - timing_start_method_f_rd_loop
            print(f"Read loop f time: {elapsed_time:.2f} seconds")

            current_datetime = datetime.now()
            print("Current date and time:", current_datetime)

        timing_end_method_f = time.time()
        elapsed_time = timing_end_method_f - timing_start_method_f
        print(f"Total time taken for method f: {elapsed_time:.2f} seconds")
        logging.info(f"Total time taken for method f: {elapsed_time:.2f} seconds")

        self.superdark = self.superdark_f

    def process_quadrant(self, read_i, file_list, n_reads_list, input_path, quadrant, sig_clip_sd_low=3.0,
                         sig_clip_sd_high=3.0):
        """
        Process a specific quadrant of the image array for a given read index.

        Parameters
        ----------
        read_i: int
            The current read index.
        file_list: list
            List of files to process.
        n_reads_list: list
            List of the number of reads for each file.
        input_path: pathlib.Path
            Path to the input files.
        quadrant: str
            The quadrant to process ('quad1', 'quad2', 'quad3', 'quad4').
        sig_clip_sd_low : float, default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high : float, default = 3.0
            Upper bound limit to filter data.

        Returns
        -------
        np.ndarray
            Processed 2048x2048 quadrant.
        """
        quad_map = {
            'quad1': (0, 2048, 0, 2048),
            'quad2': (0, 2048, 2048, 4096),
            'quad3': (2048, 4096, 0, 2048),
            'quad4': (2048, 4096, 2048, 4096)
        }

        start_row, end_row, start_col, end_col = quad_map[quadrant]
        read_i_from_all_files = np.zeros((len(file_list), 2048, 2048), dtype=np.float32)

        for file_f in range(len(file_list)):
            file_name = file_list[file_f]
            file_path = input_path.joinpath(file_name)
            if read_i < n_reads_list[file_f]:  # Changed to < to ensure it stays within bounds
                try:
                    with rdm.open(file_path) as af:
                        logging.info(f"Opening file {file_path}")
                        data = af.data
                        if isinstance(data, u.Quantity):
                            data = data.value
                        read_i_from_all_files[file_f, :, :] = data[read_i, start_row:end_row, start_col:end_col]
                except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
                    logging.warning(f"Could not open {file_path} - {e}")
            else:
                continue

        nonzero_reads = np.where(~np.all(read_i_from_all_files == 0, axis=(1, 2)))[0]
        read_i_reduced_cube = read_i_from_all_files[nonzero_reads]
        read_i_reduced_cube_no_nans = read_i_reduced_cube[~np.isnan(read_i_reduced_cube)]

        clipped_reads = sigma_clip(read_i_reduced_cube_no_nans,
                                   sigma_lower=sig_clip_sd_low,
                                   sigma_upper=sig_clip_sd_high,
                                   cenfunc=np.mean,
                                   axis=0,
                                   masked=False,
                                   copy=False)

        return np.mean(clipped_reads, axis=0)

    def assemble_quadrants(self, quad1, quad2, quad3, quad4):
        """
        Assemble the four 2048x2048 quadrants into a single 4096x4096 array.

        Parameters
        ----------
        quad1 : np.ndarray
            Upper left quadrant.
        quad2 : np.ndarray
            Upper right quadrant.
        quad3 : np.ndarray
            Lower left quadrant.
        quad4 : np.ndarray
            Lower right quadrant.

        Returns
        -------
        np.ndarray
            Combined 4096x4096 array.
        """
        upper_half = np.concatenate((quad1, quad2), axis=1)
        lower_half = np.concatenate((quad3, quad4), axis=1)
        full_array = np.concatenate((upper_half, lower_half), axis=0)
        return full_array

    def generate_outfile(self, file_permission=0o666):
        """
        Writes the superdark specified asdf outfile.

        Parameters
        ----------
        file_permission: octal string, default = 0o666
            Default file permission is rw-rw-rw- in symbolic notation meaning:
            owner, group and others have read and write permissions.
        """

        # Set reference pixel border to zero for super dark.
        # Ensure multi processing returns a full assembled super dark cube.
        self.superdark[:, :4, :] = 0.0
        self.superdark[:, -4:, :] = 0.0
        self.superdark[:, :, :4] = 0.0
        self.superdark[:, :, -4:] = 0.0

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        af.tree = {'meta': self.meta_data,
                   'data': self.superdark}
        af.write_to(self.outfile)
        os.chmod(self.outfile, file_permission)
        logging.info(f"Saved {self.outfile}")
