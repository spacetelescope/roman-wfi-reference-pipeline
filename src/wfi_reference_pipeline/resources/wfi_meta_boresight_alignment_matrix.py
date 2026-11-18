import math
from dataclasses import dataclass
from typing import ClassVar
 
import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata
 
 
@dataclass
class WFIMetaBoresightAlignmentMatrix(WFIMetadata):
    """
    Class WFIMetaBoresightAlignmentMatrix() Metadata Specific to the
    Boresight Alignment Matrix (BAM) Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first
 
    The BAM is an observatory level quantity (Roman boresight to FGS) and is
    agnostic of the detector, so instrument.detector is always written as
    NOT_CONFIGURED, which is allowed by the RAD ref_common schema.
 
    BAM specific fields
    -------------------
    v2ref: float
        V2 coordinate of the boresight in arcseconds.
    v3ref: float
        V3 coordinate of the boresight in arcseconds.
    v3ideal_y_angle: float
        Angle from the V3 axis to the Y axis in degrees, within [-360, 360].
    """
 
    # These are required reftype specific
    v2ref: float
    v3ref: float
    v3ideal_y_angle: float
 
    allow_not_configured_detector: ClassVar[bool] = True
 
    def __post_init__(self):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_BORESIGHTALIGNMENTMATRIX
        # Detector agnostic, always NOT_CONFIGURED regardless of the detector passed in.
        self.instrument_detector = constants.WFI_DETECTOR_NOT_CONFIGURED
 
        for name in ("v2ref", "v3ref", "v3ideal_y_angle"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"'{name}' must be a number, got {type(value)}.")
            if not math.isfinite(value):
                raise ValueError(f"'{name}' must be finite, got {value}.")
            setattr(self, name, float(value))
 
        if not -360.0 <= self.v3ideal_y_angle <= 360.0:
            raise ValueError(
                f"'v3_yangle' must be within [-360, 360] degrees, got {self.v3ideal_y_angle}."
            )
 
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
            # BAM meta
            'v2ref': self.v2ref,
            'v3ref': self.v3ref,
            'v3ideal_y_angle': self.v3ideal_y_angle,
        }
        return asdf_meta
    