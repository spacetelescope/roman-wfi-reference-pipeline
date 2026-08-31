import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import asdf
import numpy as np
from astropy.stats import sigma_clip
from astropy.time import Time
from roman_datamodels import dqflags

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_FGS_MASK,
    REF_TYPE_MASK,
    WFI_FRAME_TIME,
    WFI_MASK_REF_TYPES,
    WFI_MODE_WIM,
    WFI_REF_TYPES_WITHOUT_INPUT_DATA,
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
    mask_size : tuple, optional
        Expected detector dimensions.
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

        # check to see if file currently exists
        self.check_outfile()

        # Resolve data model or tree
        obj = datamodel_tree if datamodel_tree else self.populate_datamodel_tree()

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


class ReferenceTypeMask(ABC):
    """
    Base class for MASK and FGS_MASK reference files. This class supports two workflows 
    for creating a mask reference file.

    Monthly Workflow
    ----------------
    A new super dark and super rate are generated from required input files.

    Required:
        - dark_filelist
        - flat_filelist

    Weekly Workflow
    ---------------
    A new super dark is generated while an existing super rate is reused.

    Required:
        - dark_filelist
        - input_super_rate

    Parameters
    ----------
    meta_data: object
        Metadata object whose reference_type must be one of
        WFI_MASK_REF_TYPES.
    dark_filelist: list
        List of dark files used to create a super dark.
    flat_filelist: list, optional
        List of flat files used to create a super rate.
        Required for the monthly workflow.
    input_super_dark: np.ndarray; default = None
        The superdark that will be used to calculate the dark rate images / ramps.
    input_super_rate: numpy.ndarray, optional
        Existing super rate image.
        Required for the weekly workflow.
    input_user_mask: 2D integer numpy array, default = None
        A 2D data quality integer mask array to be applied to reference file.
        If either a dark or flat filelist is supplied, then this input_user_mask
        array will be added to the bad pixels identified in the darks / flats workflow.
    outfile: str, optional
        Output ASDF filename.
    clobber: bool, optional
        Overwrite an existing output file.
    mask_size: tuple, optional
        Expected detector dimensions.
    """

    def __init__(
        self,
        meta_data,
        dark_filelist=None,
        flat_filelist=None,
        input_super_dark=None,
        input_super_rate=None,
        input_user_mask=None,
        outfile=None,
        clobber=False,
        mask_size=(
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        ),
    ):

        self._validate_meta_data(meta_data)

        if not isinstance(clobber, bool):
            raise TypeError(
                "'clobber' must be a boolean."
            )

        # Validating the filelist(s) and input images
        validations = (
            (dark_filelist, "dark_filelist", self._validate_file_list),
            (flat_filelist, "flat_filelist", self._validate_file_list),
            (input_super_dark, "input_super_dark", self._validate_image),
            (input_super_rate, "input_super_rate", self._validate_image),
            (input_user_mask, "input_user_mask", self._validate_image)
        )

        for value, name, validator in validations:
            if value is not None:
                validator(value, name)

        # Setting attributes
        self.meta_data = meta_data

        self.dark_filelist = dark_filelist
        self.flat_filelist = flat_filelist

        self.super_dark = input_super_dark
        self.super_rate = input_super_rate

        self.mask_image = np.zeros((DETECTOR_PIXEL_Y_COUNT, DETECTOR_PIXEL_X_COUNT), dtype=np.uint32)
        if input_user_mask is not None:
            self.mask_image = input_user_mask

        self.outfile = outfile
        self.clobber = clobber
        self.mask_size = mask_size

        self.outdir = os.path.dirname(self.outfile)

        # Creating super darks / rates as necessary
        if self.super_dark is None and self.dark_filelist:
            self.super_dark = self._prep_super_dark(self.outdir)

        if self.super_rate is None and self.flat_filelist:
            self.super_rate = self._prep_super_rate(self.outdir)

        if self.super_dark is None and self.super_rate is None and input_user_mask is None:
            raise ValueError(
                            "Mask requires user to supply either input_user_mask, super dark, "
                            "super rate image, or dark/flat file_list."
                        )


    def _prep_super_dark(self, prep_path):
        """
        Create a super dark from the prepped self.dark_filelist files.
        This function uses the DarkPipeline super dark code. 

        Parameters
        ----------
        prep_path: str
            Path to save the super dark. Super darks are saved by default.
        """
        from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
        # Need the number of reads to run the super dark code
        nreads = self._get_nreads()

        # Setting the superdark path to be in the same dir as the prepped files
        detector = self.meta_data.instrument_detector

        superdark_filename = f"superdark_for_{self.meta_data.reference_type}_{detector}.asdf"
        self.superdark_path = os.path.join(prep_path, superdark_filename)

        logging.info("Creating super dark and writing file to %s", self.superdark_path)

        # Creating the dark pipeline object and creating the super dark
        dark_pipe = DarkPipeline(detector)
        dark_pipe.prep_superdark_file(
            short_file_list=self.dark_filelist,
            outfile=self.superdark_path,
            short_dark_num_reads=nreads,
        )

        # Return the super dark
        return self._load_superdark()


    def _get_nreads(self):
        """Using the first file in self.dark_filelist, get the number of reads in the ramp."""
        if not self.dark_filelist:
            raise TypeError("No prepped dark files found in self.dark_filelist. Cannot make superdark.")
        
        with asdf.open(self.dark_filelist[0], memmap=True) as af:
            data = af["roman"]["data"]
            dark = data.value if hasattr(data, "value") else data
            nreads = dark.shape[0]

        return nreads
    

    def _load_superdark(self):
        """Load the newly-created super dark file"""
        logging.info("Loading super dark from", self.superdark_path)

        with asdf.open(self.superdark_path, memmap=True) as af:
            data = af["roman"]["data"]
            superdark = data.value if hasattr(data, "value") else data
            return np.asarray(superdark)
    

    def _prep_super_rate(self, prep_path, sig_clip_low=3.0, sig_clip_high=3.0):
        """
        This function creates a super rate image by averaging the inputted flat rate files.

        Parameters
        ----------
        prep_path: str
            Path to save the super rate. Super rates are saved by default.
        """
        rate_images = np.zeros((len(self.flat_filelist), DETECTOR_PIXEL_Y_COUNT, DETECTOR_PIXEL_X_COUNT))

        for i, file in enumerate(self.flat_filelist):
            with asdf.open(file, memmap=True) as af:
                
                data = af["roman"]["data"]
                data = data.value if hasattr(data, "value") else data

                readtimes = [[WFI_FRAME_TIME[WFI_MODE_WIM] * t] for t in range(len(data))]

                # TODO: are we getting rate images ? 
                rate_images[i, :, :] = self._slopes_uniform_weights(data, readtimes)

        # Sigma clipping to remove cosmic rays
        clipped_rates = sigma_clip(rate_images,
                                   sigma_lower=sig_clip_low,
                                   sigma_upper=sig_clip_high,
                                   cenfunc="mean",
                                   axis=0,
                                   masked=False,
                                   copy=False)

        super_rate_image = np.nanmean(clipped_rates, axis=0)
        self._save_super_rate_image(super_rate_image, prep_path)

        return super_rate_image


    def _slopes_uniform_weights(self, d, readtimes, tensor=True):
        """
        Compute ramp slopes using uniform (read-noise-limited) weights.

        Parameters
        ----------
        input_model : RampModel
            Model containing ramps.

        Returns
        -------
        slopes : ndarray
            The slope for each pixel under uniform weighting, which is optimal
            in the read noise limit.  All flags, including saturation and
            jump, will be ignored.
        """

        # The lines below compute the weight for each resultant in the case
        # of uniform weighting (a diagonal covariance matrix consisting only
        # of read noise).

        ni = np.array([len(t) for t in readtimes])
        ti = np.array([np.mean(t) for t in readtimes])
        n = np.sum(ni)
        nt = np.sum(ni * ti)
        ntt = np.sum(ni * ti**2)
        weights = (n * ni * ti - nt * ni) / (n * ntt - nt**2)

        data = d[0] if d.ndim == 4 else d

        if tensor:
            return np.tensordot(weights, data, axes=(0, 0))

        return np.sum(weights[:, None, None] * data, axis=0)


    def _save_super_rate_image(self, super_rate_image, prep_path, file_permission=0o666):
        """
        Save the super rate image to the same path as the super dark.
        """
        detector = self.meta_data.instrument_detector

        meta_data = {'pedigree': "DUMMY",
                     'description': "Super rate file calibration product "
                                     "generated from Reference File Pipeline.",
                     'date': Time(datetime.now()),
                     'detector': detector,
                     'filelist': self.flat_filelist}

        tree = {
            "roman": {
                "meta": meta_data,
                "data": super_rate_image,
            }
        }

        super_rate_filename = f"super_rate_for_{self.meta_data.reference_type}_{detector}.asdf"
        self.super_rate_path = os.path.join(prep_path, super_rate_filename)

        af = asdf.AsdfFile()
        af.tree = tree
        af.write_to(self.super_rate_path)
        os.chmod(self.super_rate_path, file_permission)


    def _normalize_super_rate_image(self, super_rate_image):
        """
        Computes the normalized super rate image by dividing the super rate
        image by its nanmean.
        """
        logging.info("Creating the normalized super rate image")
        return super_rate_image / np.nanmean(super_rate_image)


    def _validate_meta_data(self, meta_data):
        """Validate the meta data object."""

        if not hasattr(meta_data, "reference_type"):
            raise TypeError(
                "'meta data' must contain a 'reference_type' attribute."
            )

        if meta_data.reference_type not in WFI_MASK_REF_TYPES:
            raise ValueError(
                f"Reference type '{meta_data.reference_type}' is not "
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
        expected_shape=(DETECTOR_PIXEL_Y_COUNT, DETECTOR_PIXEL_X_COUNT),
    ):
        """Validate an input image."""

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"'{image_name}' must be a numpy.ndarray."
            )

        if image.dtype != np.uint32 and image_name == "input_user_mask":
            raise TypeError(
                f"'{image_name}' must be np.uint32"
            )

        if image.ndim != 2 and image_name != "input_super_dark":
            raise ValueError(
                f"'{image_name}' must be a 2D array."
            )

        if image.shape != expected_shape and image_name != "input_super_dark":
            raise ValueError(
                f"'{image_name}' must have shape "
                f"{expected_shape}. Got {image.shape}."
            )

    def _check_outfile(self):
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
        self._check_outfile()

        if self.meta_data.reference_type == REF_TYPE_MASK:
            if not hasattr(obj, "save"):
                raise TypeError(
                    "MASK reference type requires a Roman DataModel "
                    "object with a save() method."
                )
            logging.info(
                "Writing MASK reference using Roman DataModel save()."
            )
            obj.save(self.outfile)

        elif self.meta_data.reference_type == REF_TYPE_FGS_MASK:
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
                f"Unsupported reference type '{self.meta_data.reference_type}' using ReferenceTypeMask()."
            )

        os.chmod(self.outfile, file_permission)
        logging.info(f"Saved {self.outfile}")

    @abstractmethod
    def populate_datamodel_tree(self):
        """
        Enforcing data model validation before writing file and used in schema testing.
        """
        pass
