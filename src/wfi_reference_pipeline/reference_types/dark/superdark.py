import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import asdf
from astropy.time import Time


class SuperDark(ABC):
    """
    Base class SuperDark() for all SuperDark classes
    """


    def __init__(
        self,
        input_path,
        short_dark_file_list=None,
        short_dark_num_reads=46,
        long_dark_file_list=None,
        long_dark_num_reads=98,
        outfile=None,
    ):
        """
        Parameters
        ----------
        input_path: str,
            Path to input directory where files are located.
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

        # Specify file lists.
        self.input_path = Path(input_path)  #TODO do not need input path
        self.short_dark_num_reads = short_dark_num_reads
        self.long_dark_num_reads = long_dark_num_reads

        # Initialize with short_dark_file_list and long_dark_file_list
        if short_dark_file_list and long_dark_file_list:
            self.short_dark_file_list = sorted(short_dark_file_list)
            self.long_dark_file_list = sorted(long_dark_file_list)
            self.file_list = short_dark_file_list + long_dark_file_list
        else:
            raise ValueError(
                "Invalid input combination: both 'short_dark_file_list' and "
                "'long_dark_file_list' must be provided together.")

        # Get WFIXX string
        wfixx_strings = [re.search(r'(WFI\d{2})', file).group(1) for file in self.file_list if
                         re.search(r'(WFI\d{2})', file)]
        self.wfixx_string = list(set(wfixx_strings))  # Remove duplicates if needed

        # TODO need filename to have date in YYYYMMDD format probably....need to get meta data from
        # files to populate superdark meta - what is relevant besides detector and filelist and mode?
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = str(self.wfixx_string[0] + '_superdark.asdf')

        # Make Temporary Metadata for now.  TODO - This should be gathered from files or config
        self.meta_data = {'pedigree': "DUMMY",
                          'description': "Super dark file calibration product "
                                         "generated from Reference File Pipeline.",
                          'date': Time(datetime.now()),
                          'detector': self.wfixx_string,
                          'filelist': self.file_list}

        # This is the actual superdark cube
        self.superdark = None


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

        # Enforce methods for all reference file reftype modules.
    @abstractmethod
    def generate_superdark(self):
        """
        all classes must be able to generate this
        """
        pass
