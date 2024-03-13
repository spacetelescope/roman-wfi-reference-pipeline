from dataclasses import dataclass, InitVar
from typing import Optional
import wfi_reference_pipeline.constants as constants
import logging
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaReadNoise(WFIMetadata):
    """
    Class WFIMetaReadNoise() Metadata Specific to Raed Noise Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    # These are required reftype specific
    mode: InitVar[Optional[str]] = ""
    type: InitVar[Optional[str]] = ""

    def __post_init__(self, mode, type):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_READNOISE
        if mode in constants.WFI_MODES:
            self.mode = mode
            # TODO Currently hard coding these values in, will need to evaluate later
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
            # Ref type specific meta
            'exposure': {'type': self.type,
                         'p_exptype': self.p_exptype
                        }
        }
        logging.debug(f"export_asdf_meta from WFI_MetaReadNoise {asdf_meta}")

        return asdf_meta
