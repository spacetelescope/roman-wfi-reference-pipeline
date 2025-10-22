import logging
import asdf
import yaml
from pathlib import Path

import roman_datamodels.stnode as rds
import roman_datamodels as rdm
import numpy as np

from wfi_reference_pipeline.resources.wfi_meta_darkdecaysignal import WFIMetaDarkDecay
from ..reference_type import ReferenceType

# Path to the configuration file
DARK_DECAY_SIGNAL_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "dark_decay_config.yml"
)

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
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input data intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_readnoise.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        """
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber,
            )

        # Validate metadata type
        if not isinstance(meta_data, WFIMetaDarkDecay):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaDarkDecay"
            )

        # Add default description if missing
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI Dark Decay Signal reference file."

        self.amp = None
        self.decay = None

        self.outfile = outfile
        
        self._validate_and_load_ref_type_data(ref_type_data)
        self.populate_darkdecay_arrays()

    # ----------------------------------------------------------
    # Validation and input handling
    # ----------------------------------------------------------
    def _validate_and_load_ref_type_data(self, ref_type_data):
        """
        Validate ref_type_data format and populate self.amp and self.decay.

        Allowed forms:
          1. [amp_value, decay_value] — two floats
          2. [amp_array, decay_array] — two 2D numpy arrays of identical shape
        """
        arr = np.asarray(ref_type_data, dtype=object)

        # Check dimension of input data
        if len(arr) != 2:
            raise ValueError(
                "ref_type_data must contain exactly two elements: "
                "[amplitude, decay_constant]"
            )
        amp, decay = arr[0], arr[1]

        # If both are floats
        if np.isscalar(amp) and np.isscalar(decay):
            self.amp = float(amp)
            self.decay = float(decay)
            log.info(
                f"Loaded scalar amplitude={self.amp}, decay_constant={self.decay} from ref_type_data."
            )

        # If both are 2D arrays
        elif isinstance(amp, np.ndarray) and isinstance(decay, np.ndarray):
            if amp.shape != decay.shape:
                raise ValueError(
                    "ref_type_data 2D arrays must have identical dimensions "
                    f"(got {amp.shape} vs {decay.shape})."
                )
            self.amp = amp
            self.decay = decay
            log.info(
                f"Loaded 2D amplitude and decay arrays of shape {amp.shape} from ref_type_data."
            )

        else:
            raise TypeError(
                "Invalid ref_type_data type. It must be either:\n"
                " - [float, float] for amplitude and decay constant values\n"
                " - [2D numpy array, 2D numpy array] for amplitude and decay maps"
            )

    def populate_darkdecay_arrays(self):
        """
        Ensure amplitude and decay constants are 4096x4096 arrays.
        """
        if np.isscalar(self.amp) and np.isscalar(self.decay):
            self.amp = np.full((4096, 4096), self.amp, dtype=np.float32)
            self.decay = np.full((4096, 4096), self.decay, dtype=np.float32)
            log.info("Generated 4096x4096 amplitude and decay arrays from scalar values.")

        elif isinstance(self.amp, np.ndarray) and isinstance(self.decay, np.ndarray):
            if self.amp.shape != (4096, 4096) or self.decay.shape != (4096, 4096):
                raise ValueError(
                    f"Expected 2D arrays of shape (4096,4096), got {self.amp.shape} and {self.decay.shape}."
                )
            log.info("Amplitude and decay arrays already 4096x4096 — no action needed.")
        else:
            raise TypeError("Amplitude and decay constants must be floats or 2D numpy arrays.")
    
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
        dark_decay_ref["amplitude"] = self.amp
        dark_decay_ref["amplitude_err"] = self.amp_error
        dark_decay_ref["decay_constant"] = self.decay
        dark_decay_ref["decay_const_error"] = self.decay_err
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
    
# ----------------------------------------------------------
# Standalone helper to fetch detector amplitude & decay
# ----------------------------------------------------------
def get_darkdecay_values_from_config(detector_id, config_path=DARK_DECAY_SIGNAL_CONFIG):
    """
    Retrieve amplitude and decay constant values for a given detector ID
    from the dark decay signal YAML configuration.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dark decay config file not found at {config_path}. "
            "Ensure it exists under wfi_reference_pipeline/config/."
        )

    log.info(f"Loading dark decay signal configuration from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError(f"Configuration file {config_path} is empty or invalid YAML.")

    detectors = cfg.get("detectors", [])
    match = next((d for d in detectors if d["id"] == detector_id), None)
    if match is None:
        raise KeyError(f"Detector {detector_id} not found in {config_path}.")

    amp_value = float(match["amplitude_DN"])
    decay_value = float(match["decay_constant_s"])
    return [amp_value, decay_value]