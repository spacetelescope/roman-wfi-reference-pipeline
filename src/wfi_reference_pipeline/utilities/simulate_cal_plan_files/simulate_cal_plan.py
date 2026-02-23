import subprocess
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
import yaml
from astropy.time import Time

from wfi_reference_pipeline.constants import WFI_FRAME_TIME, WFI_MODE_WIM
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

"""
Because of memory constraints for an unknown reason right now,
I am not simulating the realistic short and long darks in the calibration plan
that have 46 and 98 single read resultants, respectively.
For this development we instead set short darks to be 16 single read
resultants while we set the long darks to be 28 single read resultants.

Example useage:
from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_cal_plan import *

output_dir = "/grp/roman/RFP/DEV/sim_inflight_calplan/romanisim_darks"
config_file = "simulated_darks_config.yml"
short = ShortDarkSimulation(output_dir,
                            config_file=config_file,
                            scas=["WFI01", "WFI07"], or scas="ALL_WFI"
                            auto_run=True
                            )
"""


class BaseDarkSimulation:
    """
    Base class for short and long classes to make simulations.
    """

    def __init__(
        self,
        output_dir,
        program,
        truncate,
        start_time,
        num_exposures,
        scas="ALL_WFI",
        optical_element="F213",
        ma_table_number=9003,
        level=1,
        config_file=None,
    ):
        self.output_dir = Path(output_dir)
        self.program = program
        self.truncate = truncate
        self.start_time = Time(start_time)
        self.num_exposures = num_exposures
        self.optical_element = optical_element
        self.ma_table_number = ma_table_number
        self.level = level
        self.cal_level = "cal" if level == 2 else "uncal"

        # Get the SCAs or WFI detector numbers from scas input
        self.scas = self._parse_scas(scas)

        # Get the config files to change the dark signal properites for specific detectors
        self.config = self._load_config(config_file)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Get the WFI and SCAs to simulate
    # ---------------------------------------------------------
    def _parse_scas(self, scas):
        """
        scas: multiple inputs accepted 
        Accept:
            - "ALL_WFI"
            - ["WFI01", "WFI02", ...]
            - [1, 2, 7, 18]
        Returns:
            sorted list of integer SCA or WFI detector numbers
        """

        if scas == "ALL_WFI":
            return list(range(1, 19))
        if not isinstance(scas, (list, tuple)):
            raise ValueError("scas must be 'ALL_WFI' or a list.")

        parsed = []
        for sca in scas:
            if isinstance(sca, int):
                parsed.append(sca)
            elif isinstance(sca, str):
                sca = sca.upper()
                if sca.startswith("WFI"):
                    parsed.append(int(sca.replace("WFI", "")))
                else:
                    raise ValueError(f"Invalid SCA string: {sca}")
            else:
                raise ValueError(f"Invalid SCA type: {sca}")

        # Validate SCAs are in the range WFI01-WFI18
        for sca in parsed:
            if sca < 1 or sca > 18:
                raise ValueError(f"SCA {sca} out of valid range 1–18")

        return sorted(set(parsed))

    # ---------------------------------------------------------
    # Configuration Handling
    # ---------------------------------------------------------
    def _load_config(self, config_file):
        """
        Loading a config file from the pwd directory which contains properties to override
        dark defaults in the RFP simulate dark function.
        """
        if config_file is None:
            return None

        # If config_file is just a filename, resolve relative to this module
        config_path = Path(config_file)

        if not config_path.is_absolute():
            module_dir = Path(__file__).resolve().parent
            config_path = module_dir / config_file

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_sca_dark_params(self, sca):
        """
        Merge defaults and SCA-specific overrides.
        Returns dictionary of kwargs for simulate_dark_reads().
        """
        if self.config is None:
            return {}

        params = dict(self.config.get("defaults", {}))

        sca_overrides = self.config.get("sca_overrides", {})
        if sca in sca_overrides:
            params.update(sca_overrides[sca])

        # Optional deterministic seed per SCA
        seed = params.pop("seed", None)
        if seed is not None:
            np.random.seed(seed)

        return params

    # ---------------------------------------------------------
    # Romanisim commands and filename
    # ---------------------------------------------------------
    def _make_filename(self, exp, sca):
        """
        Need to make the output file string for romanisim that should include exposure number
        and WFI detector ID number or in romanisim SCA
        """
        exp_str = f"{exp:04d}"
        sca_str = f"wfi{sca:02d}"

        return self.output_dir / (f"r{self.program}01001001001001_"
                                  f"{exp_str}_{sca_str}_"
                                  f"{self.optical_element.lower()}_{self.cal_level}.asdf"
                                  )

    def _run_romanisim(self, filename, current_time, sca):
        """
        Need to construct the command line for romanisim-make-image
        """
        command = ["romanisim-make-image",
                   "--date", current_time.isot,
                   "--nobj", "0",
                   "--usecrds", 
                   "--sca", str(sca),
                   "--level", str(self.level),
                   "--ma_table_number", str(self.ma_table_number),
                   "--truncate", str(self.truncate),
                   str(filename),
                   ]

        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    # ---------------------------------------------------------
    # ASDF Post Processing
    # ---------------------------------------------------------
    def _post_process(self, filename, sca):
        """
        Romanisim is not simulating dark signals in WFI detectors in the romanisim-make-image command. 
        It instead is simulating F213 resultants with no sources or objects with a pre-set MA Table in
        a local clone of the romanisim repository. Need to update meta data and in the future possibly
        the filename too - reminder filename is in meta as well.
        """

        # Get the override parameters to simulate a dark cube from the RFP function
        dark_params = self._get_sca_dark_params(sca)

        dark_cube, _ = simulate_dark_reads(n_reads=self.truncate,
                                           **dark_params,
                                           )

        with asdf.open(filename, mode="rw") as af:
            #TODO update romanisim to do dark exposures or just realize noise up the ramp
            af.tree["roman"]["meta"]["instrument"]["optical_element"] = "DARK"
            #TODO make issue or fix MA Table name from reference file in romanisim
            af.tree["roman"]["meta"]["exposure"]["ma_table_name"] = "IM_DIAGNOSTIC"

            # Not using the noise from romanisim yet - need to text RFP simulated dark data first
            af.tree["roman"]["data"] = dark_cube.astype(np.uint16)
            af.update()

    # ---------------------------------------------------------
    # The main runner method.
    # ---------------------------------------------------------
    def run(self):

        time_overhead = 10.  # A general overhead time between successive dark exposures that is arbitrary.

        for sca in self.scas:
            print(f"\n=== Running SCA {sca} ===")

            current_time = self.start_time.copy()

            for exp in range(1, self.num_exposures + 1):
                filename = self._make_filename(exp, sca)

                try:
                    self._run_romanisim(filename, current_time, sca)
                    self._post_process(filename, sca)
                    print(f"✔ Created {filename.name}")

                except Exception as e:
                    print(f"✘ Exposure {exp} failed: {e}")

                # Advance time
                current_time += (
                    self.truncate * WFI_FRAME_TIME[WFI_MODE_WIM] + time_overhead
                ) * u.s


# =============================================================
# Short Dark Class
# =============================================================
class ShortDarkSimulation(BaseDarkSimulation):
    """
    For simulating the inflight calibration plan, we are going to change the actual requirement 
    to something more manageable in memory and filesize. 

    Inflight plan calls for: (26) short dark exposures with 46 single read resultants.

    Implemented here: (26) short dark exposures with 16 single read resultants. 
    """
    def __init__(self, output_dir, config_file=None, scas="ALL_WFI", num_exposures=26, auto_run=False):
        super().__init__(
            output_dir=output_dir,
            program="00444",
            truncate=16,
            start_time="2026-10-01T00:00:00",
            num_exposures=num_exposures,
            scas=scas,
            config_file=config_file,
        )

        if auto_run:
            self.run()

# =============================================================
# Long Dark Class
# =============================================================
class LongDarkSimulation(BaseDarkSimulation):
    """
    For simulating the inflight calibration plan, we are going to change the actual requirement 
    to something more manageable in memory and filesize. 

    Inflight plan calls for: (24) long dark exposures with 98 single read resultants.

    Implemented here: (24) long dark exposures with 28 single read resultants. 
    """

    def __init__(self, output_dir, config_file=None, scas="ALL_WFI", num_exposures=24, auto_run=False):
        super().__init__(
            output_dir=output_dir,
            program="00445",
            truncate=28,
            start_time="2026-10-01T00:00:00",
            num_exposures=num_exposures,
            scas=scas,
            config_file=config_file,
        )

        if auto_run:
            self.run()
