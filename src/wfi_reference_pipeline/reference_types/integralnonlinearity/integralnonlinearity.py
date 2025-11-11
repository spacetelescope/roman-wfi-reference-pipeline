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

from wfi_reference_pipeline.resources.wfi_meta_integralnonlinearity import WFIMetaINL
from ..reference_type import ReferenceType


class IntegralNonLinearity(ReferenceType):
    """
    Class IntegralNonLinearity() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_inl.asdf",
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
        if not isinstance(meta_data, WFIMetaINL):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaINL"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI gain reference file."

        self.inl_correction = ref_type_data
        self.channel_num = [i for i in range(1, 33)]
        # TODO look at references for channel id number - https://roman-docs.stsci.edu/data-handbook-home/wfi-data-format/coordinate-systems
        self.col_indices = [(i, i + 128) for i in range(0, 4096, 128)]        

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
        Build the Roman datamodel tree for the detector status reference.
        """
        try:
            # Placeholder until official datamodel exists
            inl_ref = rds.IntegralNonLinearity()
        except AttributeError:
            inl_ref = {"meta": {}, 
                       "channel_num": {},
                       "col_indices": {},
                       "num_counts_array": {}
                       }

        inl_ref["meta"] = self.meta_data.export_asdf_meta()
        inl_ref["channel_num"] = self.channel_num
        inl_ref["col_indices"] = self.col_indices
        inl_ref["num_counts_array"] = self.inl_correction

        return inl_ref
    

def make_inl_correction_array():

    N = 65536
    x = np.linspace(0, 65535, N)
    mid = (N - 1) // 2

    num_chan = 32
    arrays = []

    for _ in range(num_chan):
        # Linear slope for first half
        linear_first = np.linspace(0, 3, mid+1)

        # Low-frequency sawtooth with random phase offset (0–180 degrees)
        phase_offset_deg = np.random.uniform(0, 180)
        phase_offset_rad = np.deg2rad(phase_offset_deg)
        num_humps_saw = 1.7
        phase_saw = ((num_humps_saw * x[:mid+1] / mid) * 2 * np.pi + phase_offset_rad) % (2 * np.pi)
        saw = 5 * (phase_saw / np.pi - 1) 

        # High-frequency sine
        num_humps_sine = 4.3
        sine = np.sin(3 * np.pi * num_humps_sine * x[:mid+1] / N)

        # Combine
        y_first = linear_first + saw + sine

        # diagonal symmetry
        y_second = -y_first[::-1]
        y = np.concatenate([y_first, y_second])

        # Add Gaussian noise σ = 0.2
        noise = np.random.normal(0, 0.2, size=N)
        y_noisy = y + noise

        arrays.append(y_noisy)

    return arrays