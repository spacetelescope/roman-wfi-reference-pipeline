import logging
import asdf
import numpy as np
from astropy.time import Time
from pathlib import Path
import psutil
from datetime import datetime
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


class SuperDarkBase:
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

        # Meta data for RFP tracking and usage. Not a CRDS delivered product.
        self.meta_data = {'pedigree': "DUMMY",
                          'description': "Super dark file calibration product "
                                         "generated from Reference File Pipeline.",
                          'date': Time(datetime.now()),
                          'detector': self.wfixx_string,
                          'filelist': self.file_list}

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
