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

    self.ref_type_data: attribute;
        Class dependent variable assigned as attribute. Intended to be list of files or numpy array.
    self.ancillary: attribute;
        Other data for WFI such as filter names, frame times, WFI mode.
    self.dqflag_defs:
    """

    def __init__(self,
                 meta_data,
                 file_list=None,
                 ref_type_data=None,
                 bit_mask=None,
                 outfile=None,
                 clobber=False,
                 make_mask=False,
                 mask_size=(4096, 4096)):

        # Check to make sure ReferenceType is instantiated with one valid input.
        if file_list is None and ref_type_data is None:
            raise ValueError(
                "No data supplied to make reference file! You MUST supply 'file_list' or 'ref_type_data' parameters")
        if file_list is not None and len(file_list) > 0 and \
                ref_type_data is not None and len(ref_type_data) > 0:
            raise ValueError("Two inputs provided. Provide only one of 'file_list' or 'ref_type_data'")

        # Allow for input string use_after to be converted to astropy time object.
        if isinstance(meta_data.use_after, str):
            meta_data.use_after = Time(meta_data.use_after)

        self.meta_data = meta_data
        self.file_list = file_list
        self.outfile = outfile
        self.clobber = clobber

        #TODO fix importing dq flags from romancal
        # Load DQ flag definitions from romancal
        self.dqflag_defs = dqflags.pixel

        # TODO is this needed here or will this be reference type specific?, perhaps this hsould become an @abstractMethod ?
        if np.shape(bit_mask):
            print("Mask provided. Skipping internal mask generation.")
            self.mask = bit_mask.astype(np.uint32)
        else:
            if make_mask:
                self.mask = np.zeros(mask_size, dtype=np.uint32)
            else:
                self.mask = None

    def check_outfile(self):
        """
        Check if the output file exists, and take appropriate action.
        """

        if os.path.exists(self.outfile):
            if self.clobber:
                os.remove(self.outfile)
            else:
                raise FileExistsError(f'''{self.outfile} already exists, 
                                        and clobber={self.clobber}!''')

    def generate_outfile(self, datamodel_tree=None, file_permission=0o666):
        """
        Writes the reference file object to the specified asdf outfile.

        Parameters
        ----------
        datamodel_tree: dict, default = None
            A reftype specific dictionary built from roman data models
        file_permission: octal string, default = 0o666
            Default file permission is rw-rw-rw- in symbolic notation meaning:
            owner, group and others have read and write permissions.
        """

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
        os.chmod(self.outfile, file_permission)
        logging.info(f"Saved {self.outfile}")

    # Enforce methods for all reference file reftype modules.
    @abstractmethod
    def calculate_error(self):
        """
        If applicable, calculate error associated with reference file creation.
        """
        pass

    @abstractmethod
    def update_data_quality_array(self):
        """
        If applicable, update the reference file data quality array.
        """
        pass

    @abstractmethod
    def populate_datamodel_tree(self):
        """
        Enforcing data model validation before writing file and used in schema testing.
        """
        pass
