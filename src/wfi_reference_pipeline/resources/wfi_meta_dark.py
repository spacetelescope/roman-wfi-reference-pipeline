from dataclasses import dataclass, InitVar
from itertools import filterfalse
from typing import List, Optional
import wfi_reference_pipeline.constants  as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata

@dataclass
class WFIMetaDark(WFIMetadata):
    """
    Class WFIMetaDark() Metadata Specific to Dark Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """
    ngroups: int
    nframes: int
    groupgap: int
    ma_table_name: str
    ma_table_number: int
    type: InitVar[Optional[str]] = ""
    ref_optical_element: InitVar[Optional[List[str]]] = []

    def __post_init__(self, type, ref_optical_element):
        super().__post_init__()
        self.reference_type = constants.WFI_REF_TYPES["DARK"]
        if type in constants.WFI_MODES:
            self.type = type
            # TODO Currently hard coding these values in, will need to evaluate later
            if type == constants.WFI_MODE_WIM:
                self.p_exptype = constants.WFI_P_EXPTYPE_IMAGE
            elif type == constants.WFI_MODE_WSM:
                self.p_exptype = constants.WFI_P_EXPTYPE_GRISM + constants.WFI_P_EXPTYPE_PRISM
            else:
                raise ValueError(f"Invalid `type: {type}` for {self.reference_type}")

        if len(ref_optical_element):
            bad_elements = list(filterfalse(constants.WFI_REF_OPTICAL_ELEMENTS.__contains__, ref_optical_element))
            if not len(bad_elements):
                self.ref_optical_element = ref_optical_element
            else:
                raise ValueError(f"Invalid `ref_optical_element: {bad_elements}` for {self.reference_type}")


