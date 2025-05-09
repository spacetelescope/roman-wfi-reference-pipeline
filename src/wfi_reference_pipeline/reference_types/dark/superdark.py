import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime

import asdf
from astropy.time import Time

from wfi_reference_pipeline.constants import VIRTUAL_PIXEL_DEPTH, WFI_DETECTORS


class SuperDark(ABC):
    """
    Base class SuperDark() for all SuperDark classes
    """

    def __init__(
        self,
        short_dark_file_list,
        long_dark_file_list,
        short_dark_num_reads,
        long_dark_num_reads,
        wfi_detector_str,
        outfile=None,
    ):
        """
        Parameters
        ----------
        short_dark_file_list: list
            List of short dark exposure files. Can NOT be empty.
            This is param to use if sending in only one file list regardless of size.
        long_dark_file_list: list
            List of long dark exposure files. Can be empty.
        short_dark_num_reads: int
            Number of reads in the short dark data cubes.
        long_dark_num_reads: int
            Number of reads in the short dark data cubes.
        wfi_detector_str: str
            The FPA detector assigned number 01-18
        outfile: str, default = None
            File name written to disk.
        """


        if short_dark_num_reads < 1:
            raise ValueError(
                f"short_dark_num_reads {short_dark_num_reads} must be larger than 0"
                )

        if short_dark_num_reads > long_dark_num_reads:
            if long_dark_num_reads > 0:
                raise ValueError(
                    f"long_dark_num_reads {long_dark_num_reads} must be 0 or larger than short_dark_num_reads {short_dark_num_reads}"
                    )

        if len(short_dark_file_list) == 0:
            raise ValueError(
                "Parameter 'short_dark_file_list' can not be empty list. This is param to use if sending in only one file list regardless of size."
                )
        else:
            #verify we wre working with strings and not Paths for metadata below
            short_dark_file_list = [str(path) for path in short_dark_file_list]

        if len(long_dark_file_list) == 0:
            if long_dark_num_reads > 0:
                raise ValueError(
                    f"long_dark_num_reads {long_dark_num_reads} must be 0 if sending empty long_dark_file_list"
                    )
        else:
            #verify we wre working with posixpath
            long_dark_file_list = [str(path) for path in long_dark_file_list]


        # Specify file lists.
        self.short_dark_num_reads = short_dark_num_reads
        self.long_dark_num_reads = long_dark_num_reads

        # Initialize with short_dark_file_list and long_dark_file_list
        self.short_dark_file_list = sorted(short_dark_file_list)
        self.long_dark_file_list = sorted(long_dark_file_list)
        self.file_list = short_dark_file_list + long_dark_file_list
        self.wfi_detector_str = wfi_detector_str

        if self.wfi_detector_str not in WFI_DETECTORS:
            raise ValueError(
                f"Invalid WFI detector ID {self.wfi_detector_str}; Must be WFI01-WFI18")

        # TODO need filename to have date in YYYYMMDD format probably....need to get meta data from
        # files to populate superdark meta - what is relevant besides detector and filelist and mode?
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = str(self.wfi_detector_str) + '_superdark.asdf'

        # Make Temporary Metadata for now.  TODO - This should be gathered from files or config
        self.meta_data = {'pedigree': "DUMMY",
                          'description': "Super dark file calibration product "
                                         "generated from Reference File Pipeline.",
                          'date': Time(datetime.now()),
                          'detector': self.wfi_detector_str,
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
        self.superdark[:, :VIRTUAL_PIXEL_DEPTH, :] = 0.0
        self.superdark[:, -VIRTUAL_PIXEL_DEPTH:, :] = 0.0
        self.superdark[:, :, :VIRTUAL_PIXEL_DEPTH] = 0.0
        self.superdark[:, :, -VIRTUAL_PIXEL_DEPTH:] = 0.0

        # Despite not being a reference type,
        # keep the asdf files consistent in formatting
        af = asdf.AsdfFile()
        datamodel_tree = {'meta': self.meta_data,
                          'data': self.superdark}
        af.tree = {'roman': datamodel_tree}
        af.write_to(self.outfile)
        os.chmod(self.outfile, file_permission)
        logging.info(f"Saved {self.outfile}")

    @abstractmethod
    def generate_superdark(self):
        """
        all classes must be able to generate this
        """
        pass
