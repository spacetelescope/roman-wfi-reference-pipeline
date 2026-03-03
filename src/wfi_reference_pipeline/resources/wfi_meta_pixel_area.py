from dataclasses import InitVar, dataclass
from typing import List, Optional

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaPixelArea(WFIMetadata):
    """
    Class WFIMetaPixelArea() Metadata Specific to Pixel Area Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    pixelarea_steradians: float
    pixelarea_arcsecsq: float

    ref_optical_element: InitVar[Optional[List[str]]] = []

    def __post_init__(self, ref_optical_element):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_PIXELAREA
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
                           'optical_element': self.optical_element,
                           },
            'photometry': {'pixelarea_steradians': self.pixelarea_steradians, 
                           'pixelarea_arcsecsq': self.pixelarea_arcsecsq},
        }
        return asdf_meta
