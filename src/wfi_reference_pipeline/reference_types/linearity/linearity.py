import logging

import numpy as np
from roman_datamodels.datamodels import LinearityRefModel

from wfi_reference_pipeline.resources.wfi_meta_linearity import WFIMetaLinearity

from ..reference_type import ReferenceType


class Linearity(ReferenceType):
    """
    Class Linearity() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_linearity() creates the asdf linearity file.

    See Brandt, T. 2025 "A Classic Nonlinearity Correction Algorithm for 
    Detectors Read Out Up-the-ramp", PASP 137 125005.
    DOI 10.1088/1538-3873/ae2713
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_linearity.asdf",
            clobber=False,
            get_crds_mask=False,
            crds_context=None,  
    ):

        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_gain.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        get_crds_mask: bool; default = False
            True to retrieve the CRDS MASK reference file for meta_data.instrument_detector 
            and store its DQ array in self.crds_bad_pixel_map. See ReferenceType.
        crds_context: str; default = None
            CRDS context used with get_crds_mask, e.g. 'roman_0042.pmap'. None uses the latest.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber,
            get_crds_mask=get_crds_mask,
            crds_context=crds_context,
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaLinearity):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaLinearity"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI linearity reference file."

        logging.debug(f"Default linearity reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.lin_coeffs_array = None  # The attribute 'coeffs' in data model.

        # Module flow creating reference file
        if self.file_list:
            # Get file list properties and select data cube.
            self.num_files = len(self.file_list)
            # Implement how to derive linearity from file list.
        else:
            if not isinstance(ref_type_data, (np.ndarray)):
                raise TypeError(
                    "Input data is neither a numpy array."
                )
            dim = ref_type_data.shape
            if len(dim) == 3:
                logging.debug("The input 3D data array is now self.lin_coeffs_array.")
                self.lin_coeffs_array = ref_type_data
                logging.debug("Ready to generate reference file.")
            else:
                raise ValueError(
                    "Input data is not a valid numpy array of dimension 3."
                )
            
    def calculate_error(self):
        """
        Abstract method not applicable to Gain.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized by Gain().
        """
        pass

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """
        linearity_datamodel_tree = LinearityRefModel()
        linearity_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        linearity_datamodel_tree['coeffs'] = self.lin_coeffs_array.astype(np.float32)
        linearity_datamodel_tree["dq"] = self.dq_mask

        return linearity_datamodel_tree
