from dataclasses import InitVar, dataclass, field
from typing import List, Optional

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaEPSF(WFIMetadata):

    ref_optical_element: InitVar[Optional[List[str]]] = None

    # --- EPSF-specific fields ---
    oversample: int = 4
    spectral_type: List[str] = field(default_factory=list)
    defocus: List[int] = field(default_factory=list)
    pixel_x: List[float] = field(default_factory=list)
    pixel_y: List[float] = field(default_factory=list)

    jitter_major: float = 0.0
    jitter_minor: float = 0.0
    jitter_position_angle: float = 0.0

    def __post_init__(self, ref_optical_element):
        super().__post_init__()

        self.reference_type = constants.REF_TYPE_EPSF
        self.optical_element = []

        # Normalize
        self.spectral_type = list(self.spectral_type)
        self.defocus = list(self.defocus)
        self.pixel_x = list(self.pixel_x)
        self.pixel_y = list(self.pixel_y)

        # --- Jitter validation ---
        for name, val in {
            "jitter_major": self.jitter_major,
            "jitter_minor": self.jitter_minor,
            "jitter_position_angle": self.jitter_position_angle,
        }.items():
            if val is None:
                raise ValueError(f"{name} is required")
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number")

        if self.jitter_major < 0 or self.jitter_minor < 0:
            raise ValueError("jitter_major and jitter_minor must be non-negative")

        if not (0.0 <= self.jitter_position_angle <= 360.0):
            raise ValueError("jitter_position_angle must be in [0, 360]")

        # --- Field validation ---
        if not self.pixel_x or not self.pixel_y:
            raise ValueError("pixel_x and pixel_y are required and cannot be empty")

        if len(self.pixel_x) != len(self.pixel_y):
            raise ValueError("pixel_x and pixel_y must have the same length")

        if not self.spectral_type:
            raise ValueError("spectral_type must not be empty")

        if not self.defocus:
            raise ValueError("defocus must not be empty")

        # --- Oversample validation ---
        if not isinstance(self.oversample, int):
            raise TypeError("oversample must be an integer")

        if self.oversample <= 0:
            raise ValueError("oversample must be a positive integer")

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

            # Instrument block
            'instrument': {
                'name': self.instrument,
                'detector': self.instrument_detector,
                'optical_element': self.optical_element
            },

            # --- EPSF-specific ---
            'oversample': self.oversample,
            'spectral_type': self.spectral_type,
            'defocus': self.defocus,
            'pixel_x': self.pixel_x,
            'pixel_y': self.pixel_y,
            'jitter_major': self.jitter_major,
            'jitter_minor': self.jitter_minor,
            'jitter_position_angle': self.jitter_position_angle,
        }
        return asdf_meta
