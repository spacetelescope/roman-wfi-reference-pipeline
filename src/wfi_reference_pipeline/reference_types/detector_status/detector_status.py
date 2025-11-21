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

from wfi_reference_pipeline.resources.wfi_meta_detector_status import WFIMetaDetectorStatus
from ..reference_type import ReferenceType


class DetectorStatus(ReferenceType):
    """
    Class DetectorStatus() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.

    This reference file is much like the MA Table reference file and is a look
    up through CRDS to determine the health and status of all 18 detectors.

    Example code for file creation:

    from wfi_reference_pipeline.reference_types.detector_status.detector_status import DetectorStatus
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

    tmp = MakeDevMeta(ref_type='DETECTORSTATUS')
    rfp_det_stat = DetectorStatus(meta_data=tmp.meta_detector_status)   
    rfp_det_stat.generate_outfile()
    """

    def __init__(self,
                 meta_data,
                 outfile="roman_detector_status.asdf",
                 clobber=False
    ):
        """
        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: list[str] | None
            reference_type baes class requires an input
        outfile: str
            Output ASDF file name.
        clobber: bool
            Whether to overwrite existing ASDF file.

        Not included
        ----------
        ref_type_data: numpy array; default = None
            No data is required to make this reference file.
        bit_mask: 2D integer numpy array, default = None
            This reference file has no data quality array.
        """
        super().__init__(meta_data, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaDetectorStatus):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDETECTORSTATUS"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI detector status reference file."

        self.status_info_dict = self._make_status_info()        

        self.outfile = outfile

    def _make_status_info(self):
        """
        Make explicit dictionary for each detector to be enabled True or False
        """
        status_info = {
            "WFI01": {"enabled": True},
            "WFI02": {"enabled": True},
            "WFI03": {"enabled": True},
            "WFI04": {"enabled": True},
            "WFI05": {"enabled": True},
            "WFI06": {"enabled": True},
            "WFI07": {"enabled": True},
            "WFI08": {"enabled": True},
            "WFI09": {"enabled": True},
            "WFI10": {"enabled": True},
            "WFI11": {"enabled": True},
            "WFI12": {"enabled": True},
            "WFI13": {"enabled": True},
            "WFI14": {"enabled": True},
            "WFI15": {"enabled": True},
            "WFI16": {"enabled": True},
            "WFI17": {"enabled": True},
            "WFI18": {"enabled": True},
        }
        return status_info
    
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
            detector_status_ref = rds.DetectorStatus()
        except AttributeError:
            detector_status_ref = {"meta": {}, 
                                   "status_info": {}
                                   }

        detector_status_ref["meta"] = self.meta_data.export_asdf_meta()
        detector_status_ref["status_info"] = self.status_info_dict

        return detector_status_ref
    