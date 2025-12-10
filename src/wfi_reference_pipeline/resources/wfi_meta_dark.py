from dataclasses import InitVar, dataclass
from itertools import filterfalse
from typing import List, Optional

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaDark(WFIMetadata):
    """
    Class WFIMetaDark() Metadata Specific to Dark Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    # These are required reftype specific
    ma_table_name: str
    ma_table_number: int
    mode: InitVar[Optional[str]] = ""
    # TODO type is a reserved word in python. Can we look at type, reftype, p_exp_type, WFI_TYPE for another word/key?
    type: InitVar[Optional[str]] = ""
    ref_optical_element: InitVar[Optional[List[str]]] = []

    def __post_init__(self, mode, type, ref_optical_element):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_DARK
        if mode in constants.WFI_MODES:
            self.mode = mode
            if mode == constants.WFI_MODE_WIM:
                self.p_exptype = constants.WFI_P_EXPTYPE_IMAGE
            elif mode == constants.WFI_MODE_WSM:
                self.p_exptype = constants.WFI_P_EXPTYPE_GRISM + constants.WFI_P_EXPTYPE_PRISM
        elif len(mode):
            raise ValueError(f"Invalid `mode: {mode}` for {self.reference_type}")

        if type in constants.WFI_TYPES:
            self.type = type  # TODO follow up on type and cross referencing reftype
        elif len(type):
            raise ValueError(f"Invalid `type: {type}` for {self.reference_type}")

        if len(ref_optical_element):
            bad_elements = list(filterfalse(constants.WFI_REF_OPTICAL_ELEMENTS.__contains__, ref_optical_element))
            if not len(bad_elements):
                self.ref_optical_element = ref_optical_element
            else:
                raise ValueError(f"Invalid `ref_optical_element: {bad_elements}` for {self.reference_type}")

    def export_asdf_meta(self):
        asdf_meta = {
            # Common meta
            'reftype': self.reference_type,
            'pedigree': self.pedigree,
            'description': self.description,
            'author': self.author,
            'useafter': self.use_after,
            'telescope': self.telescope,
            'origin': self.origin,
            'instrument': {'name': self.instrument,
                           'detector': self.instrument_detector,
                           'optical_element': ','.join(self.ref_optical_element) #TODO determine how asdf validate handles multiple optical elements, comma separated?/ # Ref type specific meta
                           },
            # Ref type specific meta
            'exposure': {'ma_table_name': self.ma_table_name,
                        'ma_table_number': self.ma_table_number,
                        'type': self.type,
                        'p_exptype': self.p_exptype
                        }
        }
        return asdf_meta
