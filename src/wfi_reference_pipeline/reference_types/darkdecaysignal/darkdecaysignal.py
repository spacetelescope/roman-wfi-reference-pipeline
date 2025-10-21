import logging
import asdf
import yaml
from pathlib import Path

import roman_datamodels.stnode as rds
import roman_datamodels as rdm

from wfi_reference_pipeline.resources.darkdecaysignal import WFIMetaDarkDecay
from ..reference_type import ReferenceType

# Path to the configuration file
DARK_DECAY_SIGNAL_CONFIG = (
    Path(__file__).parent.parent.parent / "config" / "dark_decay_signal_config.yml"
).resolve()

log = logging.getLogger(__name__)


class DarkDecaySignal(ReferenceType):
    """
    The DarkDecaySignal class is responsible for creating the dark decay signal reference file
    to remove this behavior for WFI detectors.

    It reads the configuration file `dark_decay_signal_config.yml`, which contains
    the amplitude and decay constants (with uncertainties) for each detector.
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_dark_decay_signal.asdf",
        clobber=False,
    ):
        """
        Parameters
        ----------
        meta_data : dict or WFIMetaDarkDecay
            Metadata object describing the reference context.
        file_list : list[str] | None
            List of YAML config paths; defaults to dark_decay_signal_config.yml.
        outfile : str
            Output ASDF file name.
        clobber : bool
            Whether to overwrite existing file.
        """
        super().__init__(meta_data, clobber=clobber)

        # Validate metadata type
        if not isinstance(meta_data, WFIMetaDarkDecay):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDarkDecay"
            )

        # Add default description if missing
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI Dark Decay Signal reference file."

        # Determine config path
        if file_list is None or len(file_list) == 0:
            self.config_path = DARK_DECAY_SIGNAL_CONFIG
            self.file_list = [str(DARK_DECAY_SIGNAL_CONFIG)]
        else:
            self.config_path = Path(file_list[0]).resolve()
            self.file_list = [str(self.config_path)]

        self.outfile = outfile

    def get_config(self):
        """
        Load the dark decay signal YAML config file.
        Returns the dictionary of detector parameters.
        """
        log.info(f"Loading dark decay signal configuration from {self.config_path}")
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        detectors_cfg = cfg.get("detectors", [])
        return {"detectors": detectors_cfg}
    
    

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the dark decay signal reference.
        """
        try:
            # Placeholder until official datamodel exists
            dark_decay_ref = rds.DarkDecaySignalRef()
        except AttributeError:
            dark_decay_ref = {"meta": {}, "config": {}}

        dark_decay_ref["meta"] = self.meta_data.export_asdf_meta()
        dark_decay_ref["config"] = self.get_config()
        return dark_decay_ref

    def save_dark_decay_signal_file(self):
        """
        Write the dark decay signal reference ASDF file.
        """
        af = asdf.AsdfFile()
        af.tree = {"roman": self.populate_datamodel_tree()}
        log.info(f"Writing dark decay signal reference to {self.outfile}")
        af.write_to(self.outfile, overwrite=True)

    # Optional inherited stubs for completeness
    def calculate_error(self):
        return super().calculate_error()

    def update_data_quality_array(self):
        return super().update_data_quality_array()