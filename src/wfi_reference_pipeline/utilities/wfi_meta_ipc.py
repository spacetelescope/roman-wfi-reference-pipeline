from dataclasses import dataclass
from wfi_reference_pipeline.constants import WFI_REF_TYPES
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata

@dataclass
class WFIMetaIPC(WFIMetadata):
    """
    Class WFIMetaIPC() Metadata Specific to IPC Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """
    p_optical_element: str

    def __post_init__(self):
        super().__post_init__()
        self.reference_type = WFI_REF_TYPES["IPC"]