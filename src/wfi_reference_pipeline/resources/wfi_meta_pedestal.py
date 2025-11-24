from dataclasses import InitVar, dataclass
from typing import List, Optional

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata

@dataclass 
class WFIMetaPedestal(WFIMetadata):
    """
    Class WFIMetaPedestal() Metadata Specific to Fake Pedestal Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    fake_e_type: InitVar[Optional[str]] = ""

    def __post_init__(self, fake_e_type):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_PEDESTAL

        # Arbitrary stuff with fake_mode
        if fake_e_type != "good type":
            raise ValueError(f"Invalid `type: {fake_e_type}` for {self.reference_type}")


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
            # Ref file specific
            'fake_e_type' : self.fake_e_type # SYE: init-only field
        }
        return asdf_meta