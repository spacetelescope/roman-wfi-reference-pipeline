from dataclasses import InitVar, dataclass
from typing import List, Optional

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaIntegralNonLinearity(WFIMetadata):
    """
    Class WFIMetaINL() Metadata Specific to Integral Non Linearity Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first
    """
    # These are required reftype specific
    n_channels: str
    n_pixels_per_channel: str

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_INTEGRALNONLINEARITY

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
                           'detector': self.instrument_detector
                           },
            'n_channels': self.n_channels,
            'n_pixels_per_channel': self.n_pixels_per_channel,
        }
        return asdf_meta
