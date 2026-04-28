import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u

from wfi_reference_pipeline.constants import (
    SCI_PIXEL_X_COUNT,
    SCI_PIXEL_Y_COUNT,
    WFI_TYPE_IMAGE,
)
from wfi_reference_pipeline.reference_types.data_cube import DataCube
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat

from ..reference_type import ReferenceType


class Flat(ReferenceType):
    """
    Class for constructing a full focal plane flat (FlatRefModel).

    Combines:
    - Pixel flats (p-flat): per detector (18 total)
    - Large-scale flat (l-flat): global illumination structure
    """

    def __init__(
        self,
        meta_data,
        pixel_flat=None,        # pixel scale flat array
        large_flat=None,        # large scale flat array
        outfile="roman_flat.asdf",
        clobber=False,
    ):
        super().__init__(
            meta_data=meta_data,
            file_list=None,
            ref_type_data=None,
            outfile=outfile,
            clobber=clobber
        )

        # Default bit mask size of 4088x4088 for flat is size of science array
        # and must be provided if not bit_mask to instantiate properly in base class.
        if bit_mask is None:
            bit_mask = np.zeros((SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT), dtype=np.uint32)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaFlat):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaFlat"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI flat reference file."

        logging.debug(f"Default flat reference file object: {outfile} ")

        self.pflat = pixel_flat
        self.lflat = large_flat

        self.combined_flat = None
        self.combined_flat_error = None

    def combine_flat_components(self):
        """
        Combine p-flat and l-flat into a full focal plane flat.

        Algorithm TBD.
        """

        # PSEUDO:
        # 1. Stack detector-level flats into focal plane layout
        # 2. Apply L-flat correction assuming multiplicative
        # 3. Normalize
        
        wfi_fov_p_flat = self._assemble_detector_grid(self.pflat_list)

        self.combined_flat = wfi_fov_p_flat * self.lflat

        self.combined_flat /= np.mean(self.combined_flat) 

    def calculate_error(self):
        """
        Calculate error array values.
        """

    def update_data_quality_array(self):
        """
        Update data quality array bit mask with flag integer values.
        """

    def populate_datamodel_tree(self):
        """
        Flat reference file data model with combined pixel and 
        large scale flat field array.

        Drafting adding additional components of pixel and large scale flats to the data model
        that aren't required but provide tracking with RDMT for ice monitoring.
        """
        flat_tree = rds.FlatRef()
        flat_tree["meta"] = self.meta_data.export_asdf_meta()
        flat_tree["data"] = self.combined_flat.astype(np.float32)  # This is the combined P and L flat components together for romancal and CRDS
        flat_tree["err"] = self.combined_flat_error.astype(np.float32)
        flat_tree["dq"] = self.dq_mask
        flat_tree["pixel_flat"] = self.pflat.astype(np.float32)
        flat_tree["large_flat"] = self.lflat.astype(np.float32)

        return flat_tree