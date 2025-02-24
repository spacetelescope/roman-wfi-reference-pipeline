import gc
import logging
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Process, Queue, shared_memory
from queue import Empty

import asdf
import numpy as np
import psutil
from astropy import units as u

from astropy.stats import sigma_clip
from wfi_reference_pipeline.constants import (
    DARK_LONG_NUM_READS,
    DARK_SHORT_NUM_READS,
    GB,
)

from .superdark import SuperDark


# This class is used as a flag in the queue to signal a process stop
class STOPPROCESS:
    pass

# TODO - MAKE SURE SHARED MEMORY IS FREE AFTER A TERMINATION SIGNAL IS SENT! THIS SHOULDNT HAPPEN BUT IT SHOULD BE ACCOUNTED FOR!
# investigate atexit and signal handlers  (I tried quickly but ran into some issues.)


class SuperDarkDynamic(SuperDark):
    """
    Ingest raw L1 dark calibration files and average every read for as many exposures as there are available to
    create a superdark.asdf file. This file is the assumed input into the Dark() module in the RFP and the
    input used for resampling to make dark calibration reference files for any given number of MA Tables.
    """

    def __init__(
        self,
        short_dark_file_list,
        long_dark_file_list,
        short_dark_num_reads=DARK_SHORT_NUM_READS,
        long_dark_num_reads=DARK_LONG_NUM_READS,
        wfi_detector_str=None,
        outfile=None,
    ):
        """
        Parameters
        ----------
        short_dark_file_list: list
            List of short dark exposure files.
        long_dark_file_list: list
            List of long dark exposure files.
        short_dark_num_reads: int, default = 46
            Number of reads in the short dark data cubes.
        long_dark_num_reads: int, default = 98
            Number of reads in the short dark data cubes.
        outfile: str, default="roman_superdark.asdf"
            File name written to disk.
        """

        # Access methods of base class ReferenceType.
        super().__init__(
            short_dark_file_list,
            long_dark_file_list,
            short_dark_num_reads,
            long_dark_num_reads,
            wfi_detector_str=wfi_detector_str,
            outfile=outfile,
        )

    def generate_superdark(
        self,
        sig_clip_sd_low=3.0,
        sig_clip_sd_high=3.0,
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
        """
        logging.debug(f"Begin date and time: {datetime.now()}")
        self._calculated_num_reads = max(self.short_dark_num_reads, self.long_dark_num_reads) # need this check in case no long is sent in
        self.sig_clip_sd_low = sig_clip_sd_low
        self.sig_clip_sd_high = sig_clip_sd_high

        timing_start_dynamic = time.time()
        num_cores = multiprocessing.cpu_count()
        total_available_mem = psutil.virtual_memory().available


        # Calculate the required size for the shared memory cube
        element_size = np.dtype("float32").itemsize  # Size of a single float32 element in bytes
        num_elements = (self._calculated_num_reads * 4096 * 4096)  # Total number of elements in the array
        shared_mem_size = num_elements * element_size  # Total size in bytes

        available_mem = total_available_mem - shared_mem_size # This memory will be used and unavailable for other processes

        needed_mem_per_process = (shared_mem_size * 2) + (2*GB)  # 2 cubes per process from starting and sigma clip result, plus 2 gigs for processing # TODO VERIFY
        max_num_processes = available_mem // needed_mem_per_process
        max_num_processes = min(num_cores - 1, max_num_processes - 1, self._calculated_num_reads) # reserve a process for main thread with shared memory

        logging.info("STARTING SUPERDARK DYNAMIC PROCESS")
        logging.info(
            f"Number of CPU cores available:                    {num_cores}"
        )
        logging.info(
            f"Available Memory:                                 {total_available_mem} "
        )
        logging.info(
            f"                                                  {total_available_mem / GB} GB"
        )
        logging.info(
            f"Shared Memory Size:                               {shared_mem_size / GB} GB "
        )
        logging.info(
            f"Calculated Max Additional Processes:              {max_num_processes} "
        )

        print(f"Begin Multiprocessing with {max_num_processes} processes")

        try:
            shared_mem = shared_memory.SharedMemory(create=True, size=shared_mem_size)
            # create the numpy array from the allocated memory
            superdark_shared_mem = np.ndarray(
                [self._calculated_num_reads, 4096, 4096], dtype="float32", buffer=shared_mem.buf
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
            for i in range(self._calculated_num_reads):
                task_queue.put(i)
            for _ in processes:
                task_queue.put(STOPPROCESS())

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Save shared mem before closing
            self.superdark = np.copy(superdark_shared_mem)
        except Exception as e:
            logging.error(f"Error processing generate_superdark with error: {e}")
        finally:
            if shared_mem:
                 # unlink is called only once after the last instance has been closed
                shared_mem.close()
                shared_mem.unlink()

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
        superdark_shared_mem = np.ndarray([self._calculated_num_reads, 4096, 4096], dtype="float32", buffer=shared_mem.buf)  # attach a numpy array to the memory object

        # Run until we hit our STOP_PROCESS flag or queue is empty
        try:
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

                start_file = 0
                # Use this index as not all files will be used
                used_file_index = 0
                # This code segment is to initialize our cube according to size of files with this read index so we dont have
                # unused zeroes that will affect averaging.
                # Determine the number of files to process for the current read index.
                if read_index < self.short_dark_num_reads:
                    num_files_with_this_read_index = len(self.short_dark_file_list) + len(self.long_dark_file_list)
                else:
                    num_files_with_this_read_index = len(self.long_dark_file_list)
                    # Files are sorted with all shorts followed by all long files.  If the read_index is for long only, then skip the short files.
                    start_file = len(self.short_dark_file_list)
                read_index_cube = np.zeros((num_files_with_this_read_index, 4096, 4096), dtype=np.float32)

                for file_nr in range(0, num_files_with_this_read_index):
                    file_name = self.file_list[start_file + file_nr]
                    # If the file to be opened has a valid read index then open the file and
                    # get its data and increase the file counter. Separating short
                    # darks with only 46 reads from long darks with 98 reads.
                    try:
                        with asdf.open(file_name) as asdf_file:
                            if isinstance(asdf_file.tree["roman"]["data"], u.Quantity):  # Only access data from quantity object.
                                read_index_cube[used_file_index, :, :] = asdf_file.tree["roman"]["data"].value[read_index, :, :]
                            else:
                                read_index_cube[used_file_index, :, :] = asdf_file.tree["roman"]["data"][read_index, :, :]
                            used_file_index += 1
                    except (
                        FileNotFoundError,
                        IOError,
                        PermissionError,
                        ValueError,
                    ) as e:
                        logging.warning(f"    -> PID {process_name} Read {read_index}: Could not open {str(file_name)} - {e}")
                    gc.collect()
                try:
                    # TODO NOTE: may want to have option to use utilities.data_functions.get_science_pixels_cube for sigma clipping
                    clipped_reads = sigma_clip(
                        read_index_cube.astype(np.float32),
                        sigma_lower=self.sig_clip_sd_low,
                        sigma_upper=self.sig_clip_sd_high,
                        cenfunc="mean",
                        axis=0,
                        masked=False,
                        copy=False,
                    )
                except Exception as e:
                    logging.error(f"Error Sigma Clipping read index {read_index} with exception: {e}")
                # Imperative that we ignore NaN values when calculating mean
                superdark_shared_mem[read_index] = np.nanmean(clipped_reads, axis=0)
        finally:
            if shared_mem:
                # Close the local copy. This does not close the copy in the other processes
                shared_mem.close()
