import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import asdf
import numpy as np
import psutil
from astropy import units as u
from astropy.stats import sigma_clip

from .superdark import SuperDark


class SuperDarkBatches(SuperDark):
    """
    SuperDark_Batches() is a class that will ingest raw L1 dark calibration files and average every read for
    all dark exposures available to create a superdark.asdf file. This file is the input into the Dark()
    module in the RFP to create resampled dark calibration reference files for a specific MA Table.
    """

    def __init__(
        self,
        short_dark_file_list=None,
        short_dark_num_reads=46,
        long_dark_file_list=None,
        long_dark_num_reads=98,
        outfile=None,
    ):
        """
        Parameters
        ----------
        short_dark_file_list: list, default = None
            List of short dark exposure files.
        short_dark_num_reads: int, default = 46
            Number of reads in the short dark data cubes.
        long_dark_file_list: list, default = None
            List of long dark exposure files.
        long_dark_num_reads: int, default = 98
            Number of reads in the short dark data cubes.
        outfile: str, default="roman_superdark.asdf"
            File name written to disk.
        """

        # Access methods of base class ReferenceType.
        super().__init__(
            short_dark_file_list=short_dark_file_list,
            short_dark_num_reads=short_dark_num_reads,
            long_dark_file_list=long_dark_file_list,
            long_dark_num_reads=long_dark_num_reads,
            outfile=outfile,
        )

        # The attribute that contains the i'th read from all files or exposures. This is the array
        # that is sigma clipped or filtered to remove hot and dead pixels and cosmic rays.
        self.read_i_from_all_files = None
        # The array of filtered reads from all files for the i'th read of the superdark.
        self.clipped_reads = None


    def generate_superdark(
            self,
            sig_clip_sd_low=3.0,
            sig_clip_sd_high=3.0,
            short_batch_size=4,
            long_batch_size=4
        ):
        """
        This method does a file I/O open, read, and append to a temporary cube, sigma clip, and then average
        approach for every read in both short and long darks in creating the super dark cube. Starting with
        read index 0 all files that have the read index in the allowed range will be opened and the frame from
        each exposure extracted and inserted into a temporary cube representing the number of files available
        for that read.

        Parameters
        ----------
        sig_clip_sd_low: float, default = 3.0
            Lower bound limit to filter data.
        sig_clip_sd_high: float, default = 3.0
            Upper bound limit to filter data.
        short_batch_size: int, default = 4
            Number of short dark files to process in parallel at a time.
        long_batch_size: int, default = 4
            Number of long dark files to process in parallel at a time.
        """
        current_datetime = datetime.now()
        logging.info(f"Starting super dark batches at: {current_datetime}")
        timing_start_method_e = time.time()
        logging.info("Testing super dark method with file batches.")
        logging.debug(f"Memory used at start of method: {get_mem_usage():.2f} GB")

        self.superdark = np.zeros((self.long_dark_num_reads, 4096, 4096), dtype=np.float32)
        # Loop over read to construct superdark of length of long dark reads.
        # Going into each file for every i'th read or read_i index.
        for read_i in range(0, self.long_dark_num_reads):
            timing_method_file_loop_start = time.time()
            logging.debug(f"On read {read_i} of {self.long_dark_num_reads}")
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
                short_dark_results = process_files_in_batches(self.short_dark_file_list,
                                                              short_batch_size,
                                                              read_i)
                for i, result in enumerate(short_dark_results):
                    if result is not None:
                        logging.debug(f"Assigning result from short dark file to index {i} in supderdark "
                                      f"for read {read_i}")
                        self.read_i_from_all_files[i, :, :] = result

            # Need start at the short dark results to ensure correct placement and not overwrite short dark results
            # when doing long dark parallel processing.
            long_dark_results = process_files_in_batches(self.long_dark_file_list,
                                                         long_batch_size,
                                                         read_i)
            for i, result in enumerate(long_dark_results, start=len(short_dark_results)):
                if result is not None:
                    logging.debug(f"Assigning result from long dark file to index {i} in superdark"
                                  f"for read {read_i}")
                    self.read_i_from_all_files[i, :, :] = result

            timing_method_file_loop_end = time.time()
            elapsed_file_loop_time = timing_method_file_loop_end - timing_method_file_loop_start
            logging.debug(f"File loop elapsed time: {elapsed_file_loop_time}")

            timing_start_sigmaclipmean = time.time()

            if np.isnan(self.read_i_from_all_files[i, :, :]).any():
                logging.debug("NaNs found in read_i_from_all_files data cube")
            logging.debug(f"Sigma clipping reads from all files for read_i: {read_i}")
            clipped_reads = sigma_clip(self.read_i_from_all_files,
                                       sigma_lower=sig_clip_sd_low,
                                       sigma_upper=sig_clip_sd_high,
                                       cenfunc="mean",
                                       axis=0,
                                       masked=False,
                                       copy=False)

            timing_end_sigmaclipmean = time.time()
            time_sigmaclipmean = timing_end_sigmaclipmean - timing_start_sigmaclipmean
            logging.debug(f"Sigma clip and average time: {time_sigmaclipmean:.2f} seconds")

            time_fileloop_and_sigmaclipmean = timing_end_sigmaclipmean - timing_method_file_loop_start
            logging.debug(f"File loop and sigma clip and average time: {time_fileloop_and_sigmaclipmean:.2f} seconds")

            if np.isnan(clipped_reads).any():
                logging.debug("NaNs found in sigma clipped reads cube.")
            self.superdark[read_i, :, :] = np.mean(clipped_reads, axis=0)

            logging.debug(f"Memory used at end of method: {get_mem_usage():.2f} GB")
            del clipped_reads, self.read_i_from_all_files
            gc.collect()

        timing_end_method_e = time.time()
        elapsed_time = timing_end_method_e - timing_start_method_e
        logging.info(f"Total time taken for method e: {elapsed_time:.2f} seconds")


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


def get_read_from_file(file_path, read_i):
    """
    Helper function to get a read from file data cube.

    Parameters
    ----------
    file_path: Path
        Path to the file.
    read_i: int
        Read index.

    Returns
    ----------
    np.ndarray
        Array of data for the given read index.
    """

    try:
        with asdf.open(file_path) as af:
            logging.debug(f"Opening file {file_path}")
            if isinstance(af.tree['roman']['data'], u.Quantity):  # Only access data from quantity object.
                return af.tree['roman']['data'][read_i, :, :].value
            else:
                return af.tree['roman']['data'][read_i, :, :]
    except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
        logging.warning(f"Could not open {file_path} - {e}")


def process_files_in_batches(file_list, batch_size, read_i):
    """
    Processes a list of files in batches to read data for a specific read index. This function divides
    the list of files into batches, processes each batch in parallel using a ThreadPoolExecutor,
    and reads data from each file for the given read index. Results from all files are aggregated
    into a single list.

    Parameters
    ----------
    file_path: Path
        Path to the files to be processed.
    file_list: list of str
        List of file names to be processed at the file_path.
    batch_size: int
        Number of files to process in parallel at a time.
    read_i: int
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
            futures = [executor.submit(get_read_from_file, file, read_i) for file in batch]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)
    return all_results
