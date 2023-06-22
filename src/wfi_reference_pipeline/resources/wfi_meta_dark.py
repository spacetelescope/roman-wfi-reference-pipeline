from dataclasses import dataclass
from wfi_reference_pipeline.constants import WFI_REF_TYPES
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata

@dataclass
class WFIMetaDark(WFIMetadata):
    """ Metadata Specific to Dark Reference File Type """
    ngroups: int
    nframes: int
    groupgap: int
    ma_table_name: str
    ma_table_number: int
    p_optical_element: str

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = WFI_REF_TYPES["DARK"]

    # TODO remove this eventually, just here for an example
    def initialize_reference_data(self, reference):
        # TODO put code here for initializaton code from the ReferenceFile Class, Print used as temp example
        print(reference.data)