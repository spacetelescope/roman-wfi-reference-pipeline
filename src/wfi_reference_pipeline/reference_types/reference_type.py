import asdf
import logging
import os
import numpy as np
from astropy.time import Time
from romancal.lib import dqflags
from abc import ABC, abstractmethod


class ReferenceType(ABC):
    """
    Base class ReferenceType() for all reference file types.

    Returns
    -------
    self.meta_data: object;
        Reference type specific meta data object.
    self.file_list: attribute;

    self.data_array: attribute;
        Class dependent variable assigned as attribute. Intended to be list of files or numpy array.
    self.ancillary: attribute;
        Other data for WFI such as filter names, frame times, WFI mode.
    self.dqflag_defs:
    """

    def __init__(self,
                 meta_data,
                 file_list=None,
                 data_array=None,
                 bit_mask=None,
                 outfile=None,
                 clobber=False,
                 make_mask=False,
                 mask_size=(4096, 4096)):

        self.meta_data = meta_data
        # Allow for input string use_after to be converted to astropy time object.
        if isinstance(self.meta_data.use_after, str):
            self.meta_data.use_after = Time(self.meta_data.use_after)

        self.file_list = file_list
        self.data_array = data_array

        # TODO is this needed here or will this be reference type specific?, perhaps this hsould become an @abstractMethod ?
        if np.shape(bit_mask):
            print("Mask provided. Skipping internal mask generation.")
            self.mask = bit_mask.astype(np.uint32)
        else:
            if make_mask:
                self.mask = np.zeros(mask_size, dtype=np.uint32)
            else:
                self.mask = None

        self.outfile = outfile
        self.clobber = clobber

        # Load DQ flag definitions from romancal
        self.dqflag_defs = dqflags.pixel

    def check_outfile(self):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(self.outfile):
            if self.clobber:
                os.remove(self.outfile)
            else:
                raise FileExistsError(f'''{self.outfile} already exists,
                                          and clobber={self.clobber}!''')

    def generate_outfile(self, datamodel_tree=None):
        """
        Writes the reference file object to the specified asdf outfile.
        """

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
        os.chmod(self.outfile, 0o777)
        logging.info(f"Saved {self.outfile}")

    # Enforce method for all reference file reftype modules used in schema testing.
    @abstractmethod
    def populate_datamodel_tree(self):
        pass
