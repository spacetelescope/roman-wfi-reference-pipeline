from dataclasses import dataclass

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaDarkDecay(WFIMetadata):
    """
    Class WFIMetaDarkDecay() Metadata Specific to Dark Decay Signal Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_DARKDECAY

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
                           },
        }
        return asdf_meta
