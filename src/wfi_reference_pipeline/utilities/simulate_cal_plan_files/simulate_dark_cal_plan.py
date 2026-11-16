import inspect
import subprocess
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
import yaml
from astropy.time import Time

from wfi_reference_pipeline.constants import WFI_FRAME_TIME, WFI_MODE_WIM
from wfi_reference_pipeline.utilities.ma_table_handler import MATableHandler
from wfi_reference_pipeline.utilities.simulate_reads import (
    make_dark_rate_image,
    simulate_dark_reads,
)
# Parameters that define a detector's rate image (where hot/warm/dead pixels are and their rates).
RATE_IMAGE_PARAMS = set(inspect.signature(make_dark_rate_image).parameters) - {"seed"}

"""
Because of memory constraints for an unknown reason right now,
I am not simulating the realistic short and long darks in the calibration plan
that have 46 and 98 single read resultants, respectively.
For this development we instead set short darks to be 16 single read
resultants while we set the long darks to be 28 single read resultants.

Example useage:
from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_cal_plan import *

output_dir = "/grp/roman/RFP/DEV/sim_inflight_calplan/romanisim_darks"
config_file = "simulate_darks_config.yml"
weekly = WeeklyDarkSimulation(output_dir,
                              config_file=config_file,
                              scas=["WFI01", "WFI07"], or scas="ALL_WFI"
                              auto_run=True
                              )
"""


class BaseDarkSimulation:
    """
    Base class for short and long classes to make simulations.
    """

    # One MATableHandler (CRDS lookup) shared by every simulation instance.
    _ma_table_handler = None
    # Detector N's rate image seed is DEFAULT_RATE_IMAGE_SEED + N unless the config sets
    # rate_image_seed, so hot/warm/dead pixel locations are fixed per detector.
    DEFAULT_RATE_IMAGE_SEED = 1000

    def __init__(
        self,
        output_dir,
        program,
        start_time,
        num_exposures,
        scas="ALL_WFI",
        optical_element="F213",
        ma_table_number=None,  # DIAGNOSTIC = 9003, IM_135_8 = 1010
        truncate=None,  # Keep only the first N resultants (romanisim --truncate)
        level=1,
        config_file=None,
    ):
        self.output_dir = Path(output_dir)
        self.program = program
        self.start_time = Time(start_time)
        self.num_exposures = num_exposures
        self.optical_element = optical_element
        self.ma_table_number = ma_table_number
        self.truncate = truncate
        self._ma_table_config = None
        self._rate_images = {}  # SCA -> dark rate image, the same for every exposure
        self._noise_seeded_scas = set()
        self.level = level
        self.cal_level = "cal" if level == 2 else "uncal"

        # Get the SCAs or WFI detector numbers from scas input
        self.scas = self._parse_scas(scas)

        # Get the config files to change the dark signal properites for specific detectors
        self.config = self._load_config(config_file)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_ma_table_handler(cls):
        """
        Retrieve the MA table reference file from CRDS once and reuse it.
        """

        if cls._ma_table_handler is None:
            cls._ma_table_handler = MATableHandler()
        return cls._ma_table_handler

    def _get_ma_table_config(self):
        """
        Look up the read pattern for self.ma_table_number in the MA table
        reference file and apply self.truncate if set.
        Returns
        -------
        dict with:
            "read_pattern": list of lists of 1-indexed read numbers per resultant
            "n_reads": total number of reads = read_pattern[-1][-1]
            "n_resultants": len(read_pattern)
        """
        if self._ma_table_config is not None:
            return self._ma_table_config
        handler = self._get_ma_table_handler()
        try:
            read_pattern, _ = handler._get_table_specific_info(self.ma_table_number)
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(
                f"MA table {self.ma_table_number} not found in the MA table reference file"
            ) from err
        # Plain Python ints so simulate_dark_reads can index and compare them.
        read_pattern = [[int(read) for read in resultant] for resultant in read_pattern]
        if self.truncate is not None:
            read_pattern = read_pattern[: self.truncate]
        self._ma_table_config = {
            "read_pattern": read_pattern,
            "n_reads": read_pattern[-1][-1],
            "n_resultants": len(read_pattern),
        }

        return self._ma_table_config

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
        if config_file is None:
            return None

        config_path = Path(__file__).resolve().parent / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

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

        # Seeds are handled in _post_process / _get_rate_image, not passed on.
        params.pop("seed", None)
        params.pop("rate_image_seed", None)

        return params

    def _get_sca_seed(self, sca):
        """
        Read-noise seed for an SCA from the config (sca_overrides first, then defaults), or None.
        """
        if self.config is None:
            return None
        sca_overrides = self.config.get("sca_overrides", {}) or {}
        if "seed" in (sca_overrides.get(sca) or {}):
            return sca_overrides[sca]["seed"]
        return (self.config.get("defaults", {}) or {}).get("seed")
    
    def _get_rate_image_seed(self, sca):
        """
        Rate image seed for an SCA. A per-SCA rate_image_seed is used as is;
        otherwise it is (defaults rate_image_seed or DEFAULT_RATE_IMAGE_SEED) + sca.
        The same SCA always gets the same seed, and different SCAs get different seeds.
        """
        config = self.config or {}
        sca_overrides = config.get("sca_overrides", {}) or {}
        if "rate_image_seed" in (sca_overrides.get(sca) or {}):
            return int(sca_overrides[sca]["rate_image_seed"])
        base = (config.get("defaults", {}) or {}).get(
            "rate_image_seed", self.DEFAULT_RATE_IMAGE_SEED
        )
        return int(base) + sca
    
    def _get_rate_image(self, sca, dark_params):
        """
        The SCA's dark rate image: hot, warm, and dead pixel locations and rates.
        Made from the SCA's seed, so it is identical for every exposure of that SCA,
        including across the weekly and long dark simulations and across runs.
        """
        if sca not in self._rate_images:
            rate_params = {k: v for k, v in dark_params.items() if k in RATE_IMAGE_PARAMS}
            seed = self._get_rate_image_seed(sca)
            print(f"Making WFI{sca:02d} dark rate image with seed {seed}.")
            self._rate_images[sca] = make_dark_rate_image(seed=seed, **rate_params)
        return self._rate_images[sca]
    
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
        command = [
            "romanisim-make-image",
            "--date", current_time.isot,
            "--nobj", "0",
            "--usecrds",
            "--sca", str(sca),
            "--level", str(self.level),
            "--ma_table_number", str(self.ma_table_number),
        ]

        if self.truncate is not None:
            command += ["--truncate", str(self.truncate)]

        command.append(str(filename))

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

        cfg = self._get_ma_table_config()

        # Same rate image (hot/warm/dead pixel population) for every exposure of this SCA.
        rate_image = self._get_rate_image(sca, dark_params)
        # Optional read-noise seed, applied once per SCA so each exposure's noise differs.
        if sca not in self._noise_seeded_scas:
            seed = self._get_sca_seed(sca)
            if seed is not None:
                np.random.seed(seed)
            self._noise_seeded_scas.add(sca)
        # Simulate every read, then average them into resultants per the read pattern.
        # The rate image parameters are not needed once the rate image is given.
        # ni stays so the cube matches the rate image size.
        noise_params = {
            k: v for k, v in dark_params.items() if k not in RATE_IMAGE_PARAMS - {"ni"}
        }

        # Simulate every read, then average them into resultants per the read pattern.
        # Apply seeded rate image for detectors if available
        dark_cube, _ = simulate_dark_reads(
            n_reads=cfg["n_reads"],
            **dark_params,
            read_pattern=cfg["read_pattern"],
            rate_image=rate_image,
            **noise_params,
        )

        # Round and clip before the integer cast: negative values would otherwise
        # wrap around to ~65535 and fractions would be truncated.
        dark_cube = np.clip(np.round(dark_cube), 0, np.iinfo(np.uint16).max).astype(np.uint16)

        with asdf.open(filename, mode="rw") as af:
            #TODO update romanisim to do dark exposures or just realize noise up the ramp
            af.tree["roman"]["meta"]["instrument"]["optical_element"] = "DARK"
            #TODO make issue or fix MA Table name from reference file in romanisim
            #af.tree["roman"]["meta"]["exposure"]["ma_table_name"] = "IM_DIAGNOSTIC"

            # Not using the noise from romanisim yet - need to text RFP simulated dark data first
            af.tree["roman"]["data"] = dark_cube
            af.update()

    # ---------------------------------------------------------
    # The main runner method.
    # ---------------------------------------------------------
    def run(self):

        time_overhead = 10.  # A general overhead time between successive dark exposures that is arbitrary.

        # Exposure length runs to the last read in the (possibly truncated) read pattern.
        n_reads = self._get_ma_table_config()["n_reads"]
        exposure_time = n_reads * WFI_FRAME_TIME[WFI_MODE_WIM]

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

                current_time += (exposure_time + time_overhead) * u.s


# =============================================================
# Weekly Simulated Dark Class
# =============================================================
class WeeklyDarkSimulation(BaseDarkSimulation):
    """
    For simulating the inflight calibration plan, we are going to change the actual requirement 
    to something more manageable in memory and filesize. 

    Inflight plan calls for: (50) dark exposures with MA Table IM_135_8.

    Only do (25) dark exposures during GBTDS seasons. Visit ID is 2 for this.
    """
    def __init__(self, 
                 output_dir, 
                 config_file=None, 
                 scas="ALL_WFI", 
                 num_exposures=50, 
                 auto_run=False):
        
        super().__init__(
            output_dir=output_dir,
            program="00901",
            ma_table_number=1010,
            start_time="2026-12-15T00:00:00",
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

    Inflight plan calls for: (10) long dark exposures with 55 single read resultants.

    Implemented here: (10) long dark exposures with 15 single read resultants. 
    """

    def __init__(self, 
                 output_dir, 
                 config_file=None, 
                 scas="ALL_WFI", 
                 num_exposures=10, 
                 auto_run=False):
        super().__init__(
            output_dir=output_dir,
            program="00902",
            ma_table_number=9003,
            truncate=15,
            start_time="2026-12-02T00:00:00",
            num_exposures=num_exposures,
            scas=scas,
            config_file=config_file,
        )

        if auto_run:
            self.run()
