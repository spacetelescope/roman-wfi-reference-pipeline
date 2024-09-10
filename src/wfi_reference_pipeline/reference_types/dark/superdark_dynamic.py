import gc
import logging
import multiprocessing
import os
import time
from datetime import datetime
from multiprocessing import Process, Queue, shared_memory
from pathlib import Path
from queue import Empty

import asdf
import numpy as np
import psutil
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.time import Time

GB = 1024**3  # 1 GB = 1,073,741,824 bytes
MB = 1024**2  # 1 MB = 1,048,576 bytes


# This class is used as a flag in the queue to signal a process stop
class STOPPROCESS:
    pass


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
        self.clipped_reads = None
        self.read_i_from_all_files = None

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

        timing_start_getmeta = time.time()

        for file_name in self.file_list:
            file_path = self.input_path.joinpath(file_name)
            logging.debug(f"Opening file {file_path}")
            try:
                with asdf.open(file_path) as asdf_file:
                    filename = asdf_file.tree["roman"]["meta"].filename
                    read_pattern = asdf_file.tree["roman"]["meta"].exposure.read_pattern
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                print(f"Error opening file {file_path}: {e}")

            # Process each file's metadata here
            # TODO figure out what we need from each file
            logging.debug(f"File: {filename}")
            logging.debug(f"Read Pattern: {read_pattern}")

            file_name_list.append(filename)
            n_reads_list.append(
                read_pattern[-1][0]
            )  # TODO why is [0] needed to not have LNode([46])

        # Need maximum number of reads to initialize empty super dark array.
        self.max_reads = np.amax([*set(n_reads_list)])
        self.n_reads_list = n_reads_list

        # Sort file_list and n_reads_list by ascending number of reads to be used when creating super dark
        # and make file I/O more efficient/
        sorted_lists = sorted(zip(self.n_reads_list, self.file_list))
        self.n_reads_list, self.file_list = zip(*sorted_lists)

        timing_end_getmeta = time.time()
        elapsed_time = timing_end_getmeta - timing_start_getmeta
        logging.info(
            f"Total time taken to get all files meta: {elapsed_time:.2f} seconds"
        )
        logging.info(
            f"Memory used at the end of get_file_list_meta_rdmope: {get_mem_usage_gb():.2f} GB"
        )

    def make_superdark_dynamic(
        self,
        sig_clip_sd_low=3.0,
        sig_clip_sd_high=3.0,
        max_reads=98,
        n_reads_list_sorted=None,
        file_list_sorted=None,
    ):
        """
        This method calculates the largest number of processes that can be run given the system resources.
        It then takes the number of reads that will be needed and creates a worker thread queue containing these
        indexes (the final of which is followed by a stop flag for each process).  Using shared memory for the
        processess to all write to, they each pull the next available "read index" off the queue and does the following
        for only that read index:
            Checks the file to be opened has a read frame matching the read index (skip if it doesn't).
            Open the file, the data extracted for that read only and appended to a temporary cube
            representing that read for all of the files in the file list.
            That cube is then sigma clipped to remove outliers - aka cosmic rays - and averaged to produce mean pixel
            dark value for that read up the ramp.
            This clipped and averaged square is then stored in the shared area cube in its appropriate read index.

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
        logging.debug(f"Begin date and time: {datetime.now()}")

        self.sig_clip_sd_low = sig_clip_sd_low
        self.sig_clip_sd_high = sig_clip_sd_high
        self.n_reads_list = n_reads_list_sorted
        self.file_list = file_list_sorted
        self.max_reads = max_reads

        timing_start_dynamic = time.time()
        num_cores = multiprocessing.cpu_count()
        available_mem = psutil.virtual_memory().available

        needed_mem_per_process = (
            20 * GB
        )  # 20 GB needed per process, realistically this is a bit more than we need.
        max_num_processes = available_mem // needed_mem_per_process
        max_num_processes = min(num_cores, max_num_processes)

        # HERE RESIDES DEBUG CODE SETTINGS - TODO - delete after dev work complete
        # self.file_list = self.file_list[:11]
        # num_cores = 1
        # num_cores = 5

        logging.debug("STARTING SUPERDARK DYNAMIC PROCESS")
        logging.debug(
            f"Max reads per file:                                 {self.max_reads}"
        )
        logging.debug(
            f"Number of CPU cores available:                      {num_cores}"
        )
        logging.debug(
            f"Available memory:                                   {available_mem} "
        )
        logging.debug(
            f"                                                    {available_mem / GB} GB"
        )
        logging.debug(
            f"Calculated Max Processes:                           {max_num_processes} "
        )

        print(f"Begin Multiprocessing with {max_num_processes} processes")

        # Calculate the required size for the shared memory cube
        element_size = np.dtype(
            "float32"
        ).itemsize  # Size of a single float32 element in bytes
        num_elements = (
            self.max_reads * 4096 * 4096
        )  # Total number of elements in the array
        shared_mem_size = num_elements * element_size  # Total size in bytes
        shared_mem = shared_memory.SharedMemory(create=True, size=shared_mem_size)
        # create the numpy array from the allocated memory
        superdark_shared_mem = np.ndarray(
            [self.max_reads, 4096, 4096], dtype="float32", buffer=shared_mem.buf
        )
        # Initialize the shared memory to 0
        superdark_shared_mem[:] = 0

        # Create a list of processes that will pull from our queue
        task_queue = Queue()
        processes = [
            Process(target=self.read_ix_process, args=(task_queue, shared_mem.name))
            for _ in range(max_num_processes)
        ]
        for p in processes:
            p.start()

        # Add read indexes (and stop flags) to the queue for our worker tasks to pull from.
        # Add each read number to the queue to be read in by the next available process.
        for i in range(self.max_reads):
            task_queue.put(i)
        for _ in processes:
            task_queue.put(STOPPROCESS())

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Save shared mem before closing
        self.superdark = np.copy(superdark_shared_mem)
        shared_mem.close()  # cleanup
        shared_mem.unlink()  # unlink is called only once after the last instance has been closed
        timing_end_dynamic = time.time()
        elapsed_time = timing_end_dynamic - timing_start_dynamic
        logging.debug(
            f"Total time taken for SUPERDARK_DYNAMIC: {elapsed_time:.2f} seconds"
        )
        logging.debug(f"END date and time: {datetime.now()}")
        logging.info("DONE!!")
        return

    def read_ix_process(self, input_queue, shared_mem_name):
        """
        This method continually pulls next available item off of the input_queue.
        The queue will consist of read indexes that will be accessed from our long and (if they exist) short files
        Read that index from all appropriately sized files and save as a cube
        Sigma clip the cube, and take the mean value (turning it into a data square)
        Save this "slice" of data into the shared memory for the appropriate read index


        Parameters
        ----------
        input_queue: Queue();
            Queue of read indexes (or STOP_PROCESS) that this method will process from
        shared_mem_name: string;
            Identifier for the SharedMemory that this method will write to
        """

        shared_mem = shared_memory.SharedMemory(
            name=shared_mem_name
        )  # create from the existing one made by the parent process
        superdark_shared_mem = np.ndarray(
            [self.max_reads, 4096, 4096], dtype="float32", buffer=shared_mem.buf
        )  # attach a numpy array to the memory object

        # Run until we hit our STOP_PROCESS flag or queue is empty
        while True:
            try:
                # Dont wait longer than a second to get the next item off the queue
                queue_item = input_queue.get(1)
            except Empty:  # multiprocessing.Queue uses an exception template from the queue library
                logging.debug("Assuming all tasks are complete. Stopping process...")
                break
            if isinstance(queue_item, STOPPROCESS):
                print("STOP FLAG received.  Stopping process...")
                break

            read_index = queue_item
            process_name = multiprocessing.current_process().name

            # This code segment is to initialize our cube according to size of files with this read index so we dont have
            # unused zeroes that will affect averaging.
            # TODO Make this size setting better using real data (calculate number of long vs short files after sorting)
            # Long files have read index >= 46, short darks only have 0-45 reads
            if read_index < 46:
                num_files_with_this_read_index = 50  # 26 short darks + 24 long darks
            else:
                num_files_with_this_read_index = 24  # 24 long darks per detector
            read_index_cube = np.zeros(
                (num_files_with_this_read_index, 4096, 4096), dtype=np.float32
            )

            # Use this index as not all files will be used
            used_file_index = 0
            for file_nr in range(0, len(self.file_list)):
                file_name = self.file_list[file_nr]
                file_path = self.input_path.joinpath(file_name)
                if read_index < self.n_reads_list[file_nr]:
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    try:
                        with asdf.open(file_path) as asdf_file:
                            if isinstance(
                                asdf_file.tree["roman"]["data"], u.Quantity
                            ):  # Only access data from quantity object.
                                read_index_cube[used_file_index, :, :] = asdf_file.tree[
                                    "roman"
                                ]["data"].value[read_index, :, :]
                            else:
                                read_index_cube[used_file_index, :, :] = asdf_file.tree[
                                    "roman"
                                ]["data"][read_index, :, :]
                            used_file_index += 1
                    except (
                        FileNotFoundError,
                        IOError,
                        PermissionError,
                        ValueError,
                    ) as e:
                        logging.warning(
                            f"    -> PID {process_name} Read {read_index}: Could not open {str(file_path)} - {e}"
                        )
                gc.collect()

            clipped_reads = sigma_clip(
                read_index_cube,
                sigma_lower=self.sig_clip_sd_low,
                sigma_upper=self.sig_clip_sd_high,
                cenfunc=np.mean,
                axis=0,
                masked=False,
                copy=False,
            )

            superdark_shared_mem[read_index] = np.mean(clipped_reads, axis=0)
        shared_mem.close()  # cleanup after yourself (close the local copy. This does not close the copy in the other processes)

    def write_superdark(self, outfile=None):
        """
        Default conditions will write the super dark cube into an asdf
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
        print("WRITING TO SUPERDARK")
        self.superdark[:, :4, :] = 0.0
        self.superdark[:, -4:, :] = 0.0
        self.superdark[:, :, :4] = 0.0
        self.superdark[:, :, -4:] = 0.0

        meta_superdark = {
            "pedigree": "DUMMY",
            "description": "Super dark file calibration product "
            "generated from Reference File Pipeline.",
            "date": Time(datetime.now()),
            "detector": "WFO01",
            "filelist": self.file_list,
        }

        # TODO need filename to have date in YYYYMMDD format probably....need to get meta data from
        # files to populate superdark meta - what is relevant besides detector and filelist and mode?

        if outfile is None:
            outfile = Path(self.input_path) / (
                meta_superdark["detector"] + "_superdark.asdf"
            )
        # self.check_outfile(superdark_outfile) # TODO do we want a check?
        logging.info("Saving superdark asdf to disk.")

        af = asdf.AsdfFile()
        af.tree = {"meta": meta_superdark, "data": self.superdark}
        af.write_to(outfile)


def get_mem_usage_gb():
    """
    Function to return memory usage throughout module.

    Returns
    ----------
    memory_usage; float
        Memory in Gigabytes being used.

    """
    memory_usage = psutil.virtual_memory().used / GB  # in GB
    return memory_usage

def get_process_memory_usage(self):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss = memory_info.rss  # Resident Set Size: physical memory used
    vms = memory_info.vms  # Virtual Memory Size: total memory used, including swap
    return rss, vms

def log_process_memory_usage(self):
    rss, vms = self.get_process_memory_usage()
    logging.debug(
        f"Memory usage: RSS={rss / (1024 ** 2):.2f} MB, VMS={vms / (1024 ** 2):.2f} MB"
    )