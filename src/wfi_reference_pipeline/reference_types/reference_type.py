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
    WFI_MASK_REF_TYPES
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

    def generate_outfile(self, datamodel_tree=None, file_permission=0o666):
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
        # CASE 1: Roman DataModel 
        # ============================================================
        if hasattr(obj, "save"):
            logging.info("Detected Roman DataModel. Using .save() method.")
            obj.save(self.outfile)

        # ============================================================
        # CASE 2: ASDF tree / stnode - need to update to all use CASE 1 now
        # ============================================================
        else:
            logging.info("Detected ASDF tree. Using AsdfFile writer.")
            af = asdf.AsdfFile()
            af.tree = {'roman': obj}
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


class MaskBase(ABC):
    """
    Abstract base class for MASK and FGS_MASK reference files.

    Mask reference files are handled separately from ReferenceType
    because they have a specialized input model.

    Exactly one of the following input modes must be provided:

    1. An existing input mask:

        input_mask

    2. Dark and flat input file lists:

        dark_filelist + flat_filelist

    3. Pre-computed super dark and super flat images:

        super_dark + super_flat
    """

    def __init__(
        self,
        meta_data,
        dark_filelist=None,
        flat_filelist=None,
        input_mask=None,
        super_dark=None,
        super_flat=None,
        bit_mask=None,
        outfile=None,
        clobber=False,
        mask_size=(
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        ),
    ):
        self._validate_metadata(meta_data)

        self._validate_input_arguments(
            dark_filelist=dark_filelist,
            flat_filelist=flat_filelist,
            input_mask=input_mask,
            super_dark=super_dark,
            super_flat=super_flat,
        )

        if (
            hasattr(meta_data, "use_after")
            and isinstance(meta_data.use_after, str)
        ):
            meta_data.use_after = Time(meta_data.use_after)

        self.meta_data = meta_data
        self.dark_filelist = dark_filelist
        self.flat_filelist = flat_filelist
        self.input_mask = input_mask
        self.super_dark = super_dark
        self.super_flat = super_flat

        self.outfile = outfile
        self.clobber = clobber
        self.mask_size = mask_size

        # TODO: fix importing DQ flags from romancal.
        self.dqflag_defs = dqflags.pixel

        self.dq_mask = self._initialize_mask(bit_mask)

    def _validate_metadata(self, meta_data):
        """
        Validate that metadata belongs to a supported mask type.
        """
        if not hasattr(meta_data, "reference_type"):
            raise TypeError(
                "'meta_data' must have a 'reference_type' attribute."
            )

        if meta_data.reference_type not in WFI_MASK_REF_TYPES:
            raise ValueError(
                "MaskBase only supports metadata for reference types "
                f"{sorted(WFI_MASK_REF_TYPES)}. "
                f"Got '{meta_data.reference_type}'."
            )

    def _validate_input_arguments(
        self,
        dark_filelist,
        flat_filelist,
        input_mask,
        super_dark,
        super_flat,
    ):
        """
        Validate the mutually exclusive input modes.
        """

        has_input_mask = input_mask is not None

        has_file_lists = (
            dark_filelist is not None
            or flat_filelist is not None
        )

        has_super_images = (
            super_dark is not None
            or super_flat is not None
        )

        if has_file_lists:

            if (
                dark_filelist is None
                or flat_filelist is None
            ):
                raise ValueError(
                    "'dark_filelist' and 'flat_filelist' "
                    "must be provided together."
                )

            if not isinstance(dark_filelist, list):
                raise TypeError(
                    "'dark_filelist' must be of type list."
                )

            if not isinstance(flat_filelist, list):
                raise TypeError(
                    "'flat_filelist' must be of type list."
                )

            if len(dark_filelist) == 0:
                raise ValueError(
                    "'dark_filelist' must contain at least one file."
                )

            if len(flat_filelist) == 0:
                raise ValueError(
                    "'flat_filelist' must contain at least one file."
                )

        if has_super_images:

            if (
                super_dark is None
                or super_flat is None
            ):
                raise ValueError(
                    "'super_dark' and 'super_flat' "
                    "must be provided together."
                )

            self._validate_image(
                super_dark,
                "super_dark",
            )

            self._validate_image(
                super_flat,
                "super_flat",
            )

        if has_input_mask:
            self._validate_mask(input_mask)

        input_mode_count = sum(
            [
                has_input_mask,
                has_file_lists,
                has_super_images,
            ]
        )

        if input_mode_count == 0:
            raise ValueError(
                "One input mode is required. Provide exactly one of: "
                "'input_mask', "
                "'dark_filelist' + 'flat_filelist', or "
                "'super_dark' + 'super_flat'."
            )

        if input_mode_count > 1:
            raise ValueError(
                "Multiple input modes provided. Provide exactly one of: "
                "'input_mask', "
                "'dark_filelist' + 'flat_filelist', or "
                "'super_dark' + 'super_flat'."
            )

    def _validate_mask(self, input_mask):
        """
        Validate an input mask.

        The mask must be a 4096 x 4096 uint32 numpy array.
        """
        expected_shape = (
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )

        if not isinstance(input_mask, np.ndarray):
            raise TypeError(
                "'input_mask' must be a numpy.ndarray. "
                f"Got {type(input_mask)}."
            )

        if input_mask.dtype != np.uint32:
            raise ValueError(
                "'input_mask' must have dtype uint32. "
                f"Got {input_mask.dtype}."
            )

        if input_mask.ndim != 2:
            raise ValueError(
                "'input_mask' must be a 2D array. "
                f"Got {input_mask.ndim} dimensions."
            )

        if input_mask.shape != expected_shape:
            raise ValueError(
                "'input_mask' must have shape "
                f"{expected_shape}. "
                f"Got {input_mask.shape}."
            )

    def _validate_image(self, image, image_name):
        """
        Validate a super dark or super flat image.
        """
        expected_shape = (
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"'{image_name}' must be a numpy.ndarray. "
                f"Got {type(image)}."
            )

        if image.ndim != 2:
            raise ValueError(
                f"'{image_name}' must be a 2D array. "
                f"Got {image.ndim} dimensions."
            )

        if image.shape != expected_shape:
            raise ValueError(
                f"'{image_name}' must have shape "
                f"{expected_shape}. "
                f"Got {image.shape}."
            )

    def _initialize_mask(self, bit_mask):
        """
        Initialize the DQ bit mask.
        """
        if bit_mask is not None:

            if not isinstance(bit_mask, np.ndarray):
                raise TypeError(
                    "'bit_mask' should be a numpy.ndarray, "
                    f"got {type(bit_mask)}."
                )

            if bit_mask.dtype != np.uint32:
                raise ValueError(
                    "'bit_mask' must be of dtype 'uint32', "
                    f"got {bit_mask.dtype}."
                )

            if bit_mask.ndim != 2:
                raise ValueError(
                    "'bit_mask' must be 2D, "
                    f"but got {bit_mask.ndim} dimensions."
                )

            return bit_mask

        return np.zeros(
            self.mask_size,
            dtype=np.uint32,
        )

    @property
    def input_mode(self):
        """
        Return the selected input mode.
        """
        if self.input_mask is not None:
            return "input_mask"

        if (
            self.dark_filelist is not None
            and self.flat_filelist is not None
        ):
            return "file_lists"

        if (
            self.super_dark is not None
            and self.super_flat is not None
        ):
            return "super_images"

        raise RuntimeError(
            "MaskBase has no valid input mode."
        )

    def check_outfile(self):
        """
        Check if the output file exists and take appropriate action.
        """
        if self.outfile is None:
            raise ValueError(
                "Output file path 'outfile' is not specified."
            )

        if os.path.exists(self.outfile):

            if self.clobber:
                os.remove(self.outfile)

                logging.info(
                    f"Existing file '{self.outfile}' removed "
                    "due to clobber=True."
                )

            else:
                raise FileExistsError(
                    f"Output file '{self.outfile}' already exists "
                    "and clobber=False."
                )

    def generate_outfile(
        self,
        datamodel_tree=None,
        file_permission=0o666,
    ):
        """
        Write the mask reference file to the specified ASDF outfile.
        """
        if self.outfile is None:
            raise ValueError(
                "Output file path 'outfile' is not specified."
            )

        obj = (
            datamodel_tree
            if datamodel_tree is not None
            else self.populate_datamodel_tree()
        )

        self.check_outfile()

        if hasattr(obj, "save"):
            logging.info(
                "Detected Roman DataModel. Using .save() method."
            )
            obj.save(self.outfile)

        else:
            logging.info(
                "Detected ASDF tree. Using AsdfFile writer."
            )
            af = asdf.AsdfFile()
            af.tree = {"roman": obj}
            af.write_to(self.outfile)

        os.chmod(
            self.outfile,
            file_permission,
        )

        logging.info(
            f"Saved {self.outfile}"
        )

    @abstractmethod
    def calculate_error(self):
        """
        Calculate the error associated with mask creation.
        """
        pass

    @abstractmethod
    def update_data_quality_array(self):
        """
        Update the mask data quality array.
        """
        pass

    @abstractmethod
    def create_mask(self):
        """
        Create the mask from the selected input mode.
        """
        pass

    @abstractmethod
    def populate_datamodel_tree(self):
        """
        Populate the mask-specific data model.
        """
        pass

