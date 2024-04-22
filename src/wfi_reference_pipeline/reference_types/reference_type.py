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
    self.input_data: attribute;
        Class dependent variable assigned as attribute. Intended to be list of files or numpy array.
    self.meta_data: object;
        ref type specific metadata object
    self.ancillary: attribute;
        Other data for WFI such as filter names, frame times, WFI mode.
    self.dqflag_defs:
    """

    def __init__(self,
                 input_data,
                 meta_data,
                 bit_mask=None,
                 clobber=False,
                 make_mask=False,
                 mask_size=(4096, 4096)):

        self.input_data = input_data
        self.meta_data = meta_data
        # Load DQ flag definitions from romancal
        self.dqflag_defs = dqflags.pixel
        self.clobber = clobber

        # Allow for input string use_after to be converted to astropy time object.
        if isinstance(self.meta_data.use_after, str):
            self.meta_data.use_after = Time(self.meta_data.use_after)

        # TODO is this needed here or will this be reference type specific?, perhaps this hsould become an @abstractMethod ?
        if np.shape(bit_mask):
            print("Mask provided. Skipping internal mask generation.")
            self.mask = bit_mask.astype(np.uint32)
        else:
            if make_mask:
                self.mask = np.zeros(mask_size, dtype=np.uint32)
            else:
                self.mask = None

    def check_outfile(self, outfile):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(outfile):
            if self.clobber:
                os.remove(outfile)
            else:
                raise FileExistsError(f'''{outfile} already exists,
                                          and clobber={self.clobber}!''')

    def save_outfile(self, datamodel_tree=None):
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
