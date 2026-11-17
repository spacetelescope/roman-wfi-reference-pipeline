from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Union

from astropy.time import Time

from wfi_reference_pipeline.constants import (
    WFI_DETECTOR_NOT_CONFIGURED,
    WFI_DETECTORS,
    WFI_PEDIGREE,
)


@dataclass
class WFIMetadata(ABC):
    """Base Class with common Metadata to be foundation of other Reference Metadata Subclasses"""

    reference_type: str
    pedigree: str
    description: str
    author: str
    _use_after: Time
    telescope: str
    origin: str
    instrument: str
    instrument_detector: str

    # Subclasses for detector agnostic reference files set this True to allow
    # instrument_detector = WFI_DETECTOR_NOT_CONFIGURED (allowed by RAD ref_common).
    allow_not_configured_detector: ClassVar[bool] = False

    @property
    def use_after(self) -> Time:
        return self._use_after

    @use_after.setter
    def use_after(self, value: Union[str, Time]):
        self._use_after = self._convert_to_time(value)

    def __post_init__(self):
        if self.pedigree not in WFI_PEDIGREE:
            raise ValueError(
                f"Invalid pedigree value. Allowed values are {WFI_PEDIGREE}"
            )

        allowed_detectors = set(WFI_DETECTORS)
        if self.allow_not_configured_detector:
            allowed_detectors.add(WFI_DETECTOR_NOT_CONFIGURED)
        if self.instrument_detector not in allowed_detectors:
            raise ValueError(
                f"Invalid instrument_detector value. Allowed values are {allowed_detectors}"
            )

        if self._use_after is not None:
            self._use_after = self._convert_to_time(self._use_after)
        else:
            self._use_after = Time(datetime.now())

    def _convert_to_time(self, value: Union[str, Time]) -> Time:
        if isinstance(value, str):
            return Time(value)
        elif isinstance(value, Time):
            return value
        else:
            raise ValueError(
                "Invalid input for _convert_to_time, must be a string or Astropy Time object."
            )

    @abstractmethod
    def export_asdf_meta(self):
        pass
