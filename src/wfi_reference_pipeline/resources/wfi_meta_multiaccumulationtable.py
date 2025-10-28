from dataclasses import dataclass

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaMultiAccumulationTable(WFIMetadata):
    """
    Class WFIMetaMultiAccumulationTable() Metadata Specific to Aperture Correction Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_MULTIACCUMULATIONTABLE

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
            'instrument': {
                'name': self.instrument, 
                'detector': 'WFI01', # This will not be used for any selection, but it is required in the schema to write and open a valid asdf file using roman_datamodels.
                },
            # specific meta
            'prd_version': self.prd_version,
        }
        return asdf_meta
