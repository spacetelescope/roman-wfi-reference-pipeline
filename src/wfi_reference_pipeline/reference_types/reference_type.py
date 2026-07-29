import logging
import os
from abc import ABC, abstractmethod

import asdf
import numpy as np
from astropy.time import Time
from roman_datamodels import dqflags

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    WFI_REF_TYPES_WITHOUT_INPUT_DATA,
    WFI_MASK_REF_TYPES,
    REF_TYPE_FGS_MASK,
    REF_TYPE_MASK
)


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
                 mask_size=(DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
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

    def generate_outfile(self, 
                         datamodel_tree=None, 
                         file_permission=0o666):
        """
        Writes the reference file object to the specified asdf outfile.
        Supports both ASDF trees and Roman DataModel objects.

        MASK reference files use Roman DataModels and are written using
        the save() method.

        FGS_MASK reference files do not have a Roman DataModel and are
        written as ASDF trees.

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

        # Resolve data model or tree
        obj = datamodel_tree if datamodel_tree else self.populate_datamodel_tree()

        # check to see if file currently exists
        self.check_outfile()

        if self.metadata.reference_type == REF_TYPE_MASK:

            if not hasattr(obj, "save"):
                raise TypeError(
                    "MASK reference type requires a Roman DataModel "
                    "object with a save() method."
                )

            logging.info(
                "Writing MASK reference using Roman DataModel save()."
            )

            obj.save(self.outfile)

        elif self.metadata.reference_type == REF_TYPE_FGS_MASK:

            logging.info(
                "Writing FGS_MASK reference using ASDF writer."
            )

            af = asdf.AsdfFile()
            af.tree = {
                "roman": obj
            }
            af.write_to(self.outfile)

        else:
            raise ValueError(
                f"Unsupported reference type '{self.metadata.reference_type}'."
            )

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



class ReferenceTypeMask(ABC):
    """
    Base class for MASK and FGS_MASK reference files. This class supports two workflows 
    for creating a mask reference file.

    Monthly Workflow
    ----------------
    A new superdark and superslope are generated from required input files.

    Required:
        - dark_filelist
        - flat_filelist

    Weekly Workflow
    ---------------
    A new superdark is generated while an existing superslope is reused.

    Required:
        - dark_filelist
        - input_superslope

    Parameters
    ----------
    metadata : object
        Metadata object whose reference_type must be one of
        WFI_MASK_REF_TYPES.

    dark_filelist : list
        List of dark files used to create a superdark.

    flat_filelist : list, optional
        List of flat files used to create a superslope.
        Required for the monthly workflow.

    input_superslope : numpy.ndarray, optional
        Existing superslope image.
        Required for the weekly workflow.

    outfile : str, optional
        Output ASDF filename.

    clobber : bool, optional
        Overwrite an existing output file.

    mask_size : tuple, optional
        Expected detector dimensions.
    """

    def __init__(
        self,
        metadata,
        dark_filelist,
        flat_filelist=None,
        input_superslope=None,
        outfile=None,
        clobber=False,
        mask_size=(
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        ),
    ):

        self._validate_metadata(metadata)

        self._validate_file_list(
            dark_filelist,
            "dark_filelist",
        )

        self._validate_mask_size(mask_size)

        if not isinstance(clobber, bool):
            raise TypeError(
                "'clobber' must be a boolean."
            )

        monthly = flat_filelist is not None
        weekly = input_superslope is not None

        if monthly == weekly:
            raise ValueError(
                "Specify exactly one workflow:\n"
                "  Monthly : flat_filelist must be provided\n"
                "  Weekly  : input_superslope must be provided"
            )

        self.workflow = "monthly" if monthly else "weekly"

        if monthly:
            self._validate_file_list(
                flat_filelist,
                "flat_filelist",
            )

        if weekly:
            self._validate_image(
                input_superslope,
                "input_superslope",
                mask_size,
            )

        self.metadata = metadata

        self.dark_filelist = dark_filelist
        self.flat_filelist = flat_filelist
        self.input_superslope = input_superslope

        self.outfile = outfile
        self.clobber = clobber
        self.mask_size = mask_size

        self.dqflag_defs = dqflags.pixel

    def _validate_metadata(self, metadata):
        """Validate the metadata object."""

        if not hasattr(metadata, "reference_type"):
            raise TypeError(
                "'metadata' must contain a 'reference_type' attribute."
            )

        if metadata.reference_type not in WFI_MASK_REF_TYPES:
            raise ValueError(
                f"Reference type '{metadata.reference_type}' is not "
                "supported by MaskBase."
            )


    def _validate_file_list(self, file_list, name):
        """Validate a file list."""

        if not isinstance(file_list, list):
            raise TypeError(
                f"'{name}' must be a list."
            )

        if len(file_list) == 0:
            raise ValueError(
                f"'{name}' must contain at least one file."
            )

        if not all(
            isinstance(filename, str)
            for filename in file_list
        ):
            raise TypeError(
                f"'{name}' must contain only strings."
            )

    def _validate_image(
        self,
        image,
        image_name,
        expected_shape,
    ):
        """Validate an input image."""

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"'{image_name}' must be a numpy.ndarray."
            )

        if image.ndim != 2:
            raise ValueError(
                f"'{image_name}' must be a 2D array."
            )

        if image.shape != expected_shape:
            raise ValueError(
                f"'{image_name}' must have shape "
                f"{expected_shape}. Got {image.shape}."
            )

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

    def generate_outfile(self, 
                         datamodel_tree=None, 
                         file_permission=0o666):
        """
        Writes the reference file object to the specified asdf outfile.
        Supports both ASDF trees and Roman DataModel objects.

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

        # Resolve data model or tree
        obj = datamodel_tree if datamodel_tree else self.populate_datamodel_tree()

        # check to see if file currently exists
        self.check_outfile()

        # ============================================================
        # CASE 1: For Mask Ref Type which has a data model
        # ============================================================
        if hasattr(obj, "save"):
            logging.info("Detected Roman DataModel. Using .save() method.")
            obj.save(self.outfile)

        # ============================================================
        # CASE 2: For FGSMask Ref Type which DOES NOT have a data model
        # ============================================================
        else:
            logging.info("Detected ASDF tree. Using AsdfFile writer.")
            af = asdf.AsdfFile()
            af.tree = {'roman': obj}
            af.write_to(self.outfile)

        os.chmod(self.outfile, file_permission)
        logging.info(f"Saved {self.outfile}")

    @abstractmethod
    def populate_datamodel_tree(self):
        """
        Enforcing data model validation before writing file and used in schema testing.
        """
        pass