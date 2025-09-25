import logging
import os
from abc import ABC, abstractmethod

import asdf
import numpy as np
from astropy.time import Time
from romancal.lib import dqflags

from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT, WFI_REF_TYPES_WITHOUT_INPUT_DATA


class ReferenceType(ABC):
    """
    Base class ReferenceType() for all reference file types.

    Parameters
    ----------
    meta_data : object
        Reference type specific meta data object.
    file_list : list or None, optional
        List of files to be used for creating the reference file.
    ref_type_data : numpy.ndarray or None, optional
        Data array for the reference type.
    bit_mask : numpy.ndarray or None, optional
        Bit mask array corresponding to the data. If provided, it should match the shape of `ref_type_data`.
    outfile : str or None, optional
        Path to the output file where the reference data will be saved.
    clobber : bool, optional
        If True, overwrites the existing outfile without warning.
    """

    def __init__(self,
                 meta_data,
                 file_list=None,
                 ref_type_data=None,
                 bit_mask=None,
                 outfile=None,
                 clobber=False,
                 mask_size=(4096, 4096)
                 ):

        have_file_list = False
        have_ref_type_data = False
        have_input = False
        if file_list is not None:
            if not isinstance(file_list, list):
                raise ValueError("'file_list' must be of type list")
            if len(file_list) > 0:
                have_file_list = True
                have_input = True
        if ref_type_data is not None and len(ref_type_data) > 0:
            have_ref_type_data = True
            have_input = True

        # Check to make sure ReferenceType is instantiated with one valid input.
        # some ref types require no input data. see constants.WFI_REF_TYPES_WITHOUT_DATA for list of those reference types
        if have_file_list and have_ref_type_data:
            raise ValueError("Two inputs provided. Provide only one of 'file_list' or 'ref_type_data'")
        if not have_input and meta_data.reference_type not in WFI_REF_TYPES_WITHOUT_INPUT_DATA:
            raise ValueError(f"Reference File type {meta_data.reference_type} requires input data in the form of a file_list or ref_type_data.")



        # Allow for input string use_after to be converted to astropy time object.
        if hasattr(meta_data, "use_after") and isinstance(meta_data.use_after, str):
            meta_data.use_after = Time(meta_data.use_after)

        self.meta_data = meta_data
        self.file_list = file_list
        self.outfile = outfile
        self.clobber = clobber
        self.mask_size = mask_size

        #TODO fix importing dq flags from romancal
        # Load DQ flag definitions from romancal
        self.dqflag_defs = dqflags.pixel

        # Handle bit_mask initialization
        self.dq_mask = self._initialize_mask(bit_mask)

    def _initialize_mask(self, bit_mask):
        """
        Initialize the bit mask based on provided parameters.

        Parameters
        ----------
        bit_mask : numpy.ndarray or None
            Provided 2D bit mask array.

        Returns
        -------
        numpy.ndarray
            Initialized mask array with shape (mask_size[0], mask_size[1]).
        """
        if bit_mask is not None:
            if not isinstance(bit_mask, np.ndarray):
                raise TypeError(f"'bit_mask' should be a numpy.ndarray, got {type(bit_mask)}.")
            if bit_mask.dtype != np.uint32:
                raise ValueError(f"'bit_mask' must be of dtype 'uint32', got {bit_mask.dtype}.")
            if bit_mask.ndim != 2:
                raise ValueError(f"'bit_mask' must be 2D, but got {bit_mask.ndim} dimensions.")
            return bit_mask
        else:
            return np.zeros(self.mask_size, dtype=np.uint32)

    def check_outfile(self):
        """
        Check if the output file exists, and take appropriate action.
        """
        if self.outfile is None:
            raise ValueError("Output file path 'outfile' is not specified.")

        if os.path.exists(self.outfile):
            if self.clobber:
                os.remove(self.outfile)
                logging.info(f"Existing file '{self.outfile}' removed due to clobber=True.")
            else:
                raise FileExistsError(
                    f"Output file '{self.outfile}' already exists and clobber=False."
                )

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
        if self.outfile is None:
            raise ValueError("Output file path 'outfile' is not specified.")

        # Use datamodel tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}

        # check to see if file currently exists
        self.check_outfile()

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
