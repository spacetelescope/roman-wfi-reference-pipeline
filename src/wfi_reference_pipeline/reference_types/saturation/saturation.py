import logging
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u
from wfi_reference_pipeline.resources.wfi_meta_saturation import WFIMetaSaturation

from ..reference_type import ReferenceType


class Saturation(ReferenceType):
    """
    Class Saturation() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    Saturation() creates the saturation reference file using roman data models
    and has all necessary meta and matching criteria for delivery to CRDS.

    Example file creation commands:
    saturation_obj = Saturation(meta_data, ref_type_data=test_data_array)
    saturation_obj.make_saturation_image()
    saturation_obj.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_saturation.asdf",
        clobber=False,
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_saturation.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType.
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaSaturation):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaSaturation."
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI saturation reference file."

        logging.debug(f"Default saturation reference file object: {outfile}.")

        # Initialize attributes.
        self.saturation_image = None

        # Module flow creating reference file.
        if not isinstance(ref_type_data, (np.ndarray, u.Quantity)):
            raise TypeError(
                "Input data is neither a numpy array nor a Quantity object."
            )
        if isinstance(
            ref_type_data, u.Quantity
        ):  # Only access data from quantity object.
            ref_type_data = ref_type_data.value
            logging.debug("Quantity object detected. Extracted data values.")

        dim = ref_type_data.shape
        if len(dim) == 2:
            logging.debug("The input 2D data array is now self.saturation_image.")
            self.saturation_image = ref_type_data.astype(np.float32)
            logging.debug("Ready to generate reference file.")
        else:
            raise ValueError(
                "Input data is not a valid numpy array of dimension 2."
            )

    def make_saturation_image(self, saturation_threshold=55000):
        """
        Make the saturation image a uniform array of a certain threshold value.

        Parameters
        ----------
        saturation_threshold: float; default = 55000
            The saturated level for development of romancal.
        """

        self.saturation_image = saturation_threshold * np.ones((4096, 4096), dtype=np.float32)

    def calculate_error(self):
        """
        Abstract method not applicable to Saturation.
        """

        pass

    def update_data_quality_array(self, bad_saturation_threshold=64000):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        bad_saturation_threshold: float; default = 64000
            Limit to update saturation reference file mask with no saturation check dq flag.
        """

        no_saturated_check_array = np.where(self.saturation_image >= bad_saturation_threshold)

        logging.info("Flagging no saturation check DQ array.")
        # Locate bad saturated pixels with no saturation check
        self.mask[self.saturation_image > no_saturated_check_array] += self.dqflag_defs['NO_SAT_CHECK']

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the saturation object from the data model.
        saturation_datamodel_tree = rds.SaturationRef()
        saturation_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        saturation_datamodel_tree["data"] = self.saturation_image * u.DN
        saturation_datamodel_tree["dq"] = self.mask

        return saturation_datamodel_tree