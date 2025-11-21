import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
import roman_datamodels as rdm
import os
import yaml
import crds
from crds.client import api
import subprocess
import shutil
from pathlib import Path


import roman_datamodels.stnode as rds

from wfi_reference_pipeline.resources.wfi_meta_pedestal import WFIMetaPedestal
from ..reference_type import ReferenceType


class Pedestal(ReferenceType):
    """
    Class Pedestal() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_pedestal.asdf",
            clobber=False,
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
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
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
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaPedestal):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaPEDESTAL"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI pedestal reference file."

        self.pedestal = ref_type_data    

        self.outfile = outfile

    def calculate_error(self):
        """
        Abstract method not applicable.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the pedestal reference.
        """
        try:
            # Placeholder until official datamodel exists
            ped_ref = rds.Pedestal()
        except AttributeError:
            ped_ref = {"meta": {}, 
                       "data": {},
                       "dq": {}
                       }

        ped_ref["meta"] = self.meta_data.export_asdf_meta()
        ped_ref["data"] = self.pedestal
        ped_ref["dq"] = np.zeros((4096, 4096), dtype=np.uint16)

        return ped_ref
    

def make_bias_array():

    arr = np.random.uniform(0.2, 0.8, size=(4096, 4096))

    return arr