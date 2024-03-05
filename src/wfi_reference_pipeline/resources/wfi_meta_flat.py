from dataclasses import dataclass, InitVar
from typing import List, Optional
import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaFlat(WFIMetadata):
    """
    Class WFIMetaFlat() Metadata Specific to Flat Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    # These are required reftype specific
    ref_optical_element: InitVar[Optional[List[str]]] = []

    def __post_init__(self, ref_optical_element):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_FLAT
        self.optical_element = ref_optical_element

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
                           'optical_element': self.optical_element
                           },
        }
        return asdf_meta
