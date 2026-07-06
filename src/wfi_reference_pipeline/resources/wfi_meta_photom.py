from dataclasses import InitVar, dataclass

import wfi_reference_pipeline.constants as constants
from wfi_reference_pipeline.resources.wfi_metadata import WFIMetadata


@dataclass
class WFIMetaPhotom(WFIMetadata):
    """
    Class WFIMetaPhotom() Metadata Specific to Photom Reference File Type
    inherits WFIMetadata
    All Fields are required and positional with base class fields first

    """

    # These are required reftype specific
    # Setting the initial gain values to 1
    ref_median_gain: InitVar[float] = 1.0
    ref_sigma_gain: InitVar[float] = 1.0

    def __post_init__(self, ref_median_gain, ref_sigma_gain):
        super().__post_init__()
        self.reference_type = constants.REF_TYPE_PHOTOM
        self.median_gain = ref_median_gain
        self.sigma_gain = ref_sigma_gain

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
                           'median_gain': self.median_gain,
                           'sigma_gain': self.sigma_gain
                           },
        }
        return asdf_meta
