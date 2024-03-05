from dataclasses import dataclass
import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaReferencePixel(WFIMetadata):
    """
    Class WFIMetaReferencePixel() Metadata Specific to ReferencePixel Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """
    # These are required reftype specific
    input_units: str
    output_units: str

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_REFPIX

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
            # Ref type specific meta
            'input_units': self.input_units,
            'output_units': self.output_units,
        }
        return asdf_meta
