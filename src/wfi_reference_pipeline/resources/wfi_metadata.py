from abc import ABC, abstractmethod
from astropy.time import Time
from dataclasses import dataclass
from datetime import datetime
from wfi_reference_pipeline.constants import WFI_DETECTORS, WFI_PEDIGREE


@dataclass
class WFIMetadata(ABC):
    """ Base Class with common Metadata to be foundation of other Reference Metadata Subclasses """
    reference_type: str
    pedigree: str
    description: str
    author: str
    use_after: Time
    telescope: str
    origin: str
    instrument: str
    instrument_detector: str


    def __post_init__(self):
        if self.pedigree not in WFI_PEDIGREE:
            raise ValueError(f"Invalid pedigree value. Allowed values are {WFI_PEDIGREE}")

        if self.instrument_detector not in WFI_DETECTORS:
            raise ValueError(f"Invalid instrument_detector value. Allowed values are {WFI_DETECTORS}")

        if isinstance(self.use_after, str):
            if len(self.use_after):
                self.use_after = Time(self.use_after)
            else:
                self.use_after = Time(datetime.now())

    @abstractmethod
    def export_asdf_meta(self):
        pass


