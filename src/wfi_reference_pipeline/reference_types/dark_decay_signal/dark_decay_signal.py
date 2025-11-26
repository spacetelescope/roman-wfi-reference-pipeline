import logging

import roman_datamodels.stnode as rds

from wfi_reference_pipeline.resources.wfi_meta_dark_decay_signal import (
    WFIMetaDarkDecaySignal,
)

from ..reference_type import ReferenceType

log = logging.getLogger(__name__)


class DarkDecaySignal(ReferenceType):
    """
    Creates a Roman WFI Dark Decay Signal reference file that stores
    amplitude and time constant for all detectors

    No array maps are created — this is a detector-level table from T. Brandt et al 2025.

    Example code to generate file:
    from wfi_reference_pipeline.reference_types.dark_decay_signal.dark_decay_signal import DarkDecaySignal
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

    tmp = MakeDevMeta(ref_type='DARKDECAYSIGNAL')
    rfp_dark_decay = DarkDecaySignal(meta_data=tmp.meta_dark_decay_signal)
    rfp_dark_decay.generate_outfile()
    """

    def __init__(self,
                 meta_data,
                 outfile="roman_dark_decay_signal.asdf",
                 clobber=False
    ):
        """
        Parameters
        ----------
        meta_data: dict
            Must include a key like {"detector": "WFI01"} but for this reference file it has
            all data for every detector. WFI01 is the default.
        outfile: str
            Output ASDF file name.
        clobber: bool
            Whether to overwrite existing ASDF file.

        Not included
        ----------
        file_list: list[str] | None
        ref_type_data: numpy array; default = None
        bit_mask: 2D integer numpy array, default = None
        """
        super().__init__(meta_data, outfile=outfile, clobber=clobber)
        

        # Metadata validation
        if not isinstance(meta_data, WFIMetaDarkDecaySignal):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDarkDecaySignal"
            )

        # Default description
        if not getattr(self.meta_data, "description", ""):
            self.meta_data.description = (
                "Roman WFI Dark Decay Signal reference file for each detector table."
            )

        self.outfile = outfile

    def populate_datamodel_tree(self):
        """
        Create the datamodel tree to be written to ASDF.
        """
        try:
            # TODO for when official datamodel exists
            dark_decay_ref = rds.DarkDecaySignalRef()
        except AttributeError:
            dark_decay_ref = {
                             "meta": {}, 
                             "decay_table": {}
                             }

        dark_decay_ref["meta"] = self.meta_data.export_asdf_meta()
        dark_decay_ref["decay_table"] = DARK_DECAY_TABLE

        return dark_decay_ref
    
    
    def calculate_error(self):
        """
        Abstract method not utilized.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass
    
# =====================================================================
#  Static dark decay table with values derived from T. Brandt et al. 2025
#  Provide URL or DOI
# =====================================================================
DARK_DECAY_TABLE = {
    "WFI01": {"amplitude": 0.15, "time_constant": 24.6},
    "WFI02": {"amplitude": 0.43, "time_constant": 22.1},
    "WFI03": {"amplitude": 0.50, "time_constant": 23.1},
    "WFI04": {"amplitude": 0.36, "time_constant": 19.6},
    "WFI05": {"amplitude": 0.42, "time_constant": 28.8},
    "WFI06": {"amplitude": 0.61, "time_constant": 22.1},
    "WFI07": {"amplitude": 0.22, "time_constant": 20.2},
    "WFI08": {"amplitude": 0.36, "time_constant": 31.4},
    "WFI09": {"amplitude": 0.48, "time_constant": 21.9},
    "WFI10": {"amplitude": 0.19, "time_constant": 22.1},
    "WFI11": {"amplitude": 0.37, "time_constant": 23.3},
    "WFI12": {"amplitude": 0.39, "time_constant": 29.5},
    "WFI13": {"amplitude": 0.26, "time_constant": 27.5},
    "WFI14": {"amplitude": 0.40, "time_constant": 25.4},
    "WFI15": {"amplitude": 0.42, "time_constant": 23.2},
    "WFI16": {"amplitude": 0.36, "time_constant": 22.7},
    "WFI17": {"amplitude": 0.67, "time_constant": 25.2},
    "WFI18": {"amplitude": 0.53, "time_constant": 21.3},
}


# =====================================================================
#  Helper function to provide quick lookups and cross checks
# =====================================================================
def get_darkdecay_values(detector_id):
    """
    Return [amplitude, time_constant] for a detector.
    """
    try:
        entry = DARK_DECAY_TABLE[detector_id]
    except KeyError:
        raise KeyError(f"Detector '{detector_id}' not found in DARK_DECAY_TABLE.")

    return [entry["amplitude"], entry["time_constant"]]
