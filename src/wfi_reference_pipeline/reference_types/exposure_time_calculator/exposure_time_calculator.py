
import os
import shutil
import subprocess
from pathlib import Path

import crds
import numpy as np
import roman_datamodels as rdm
import roman_datamodels.stnode as rds
import yaml
from crds.client import api

from wfi_reference_pipeline.resources.wfi_meta_exposure_time_calculator import (
    WFIMetaETC,
)

from ..reference_type import ReferenceType

ETC_FORM = (Path(__file__).parent / "exposure_time_calculator_form.yml").resolve()


class ExposureTimeCalculator(ReferenceType):
    """
    Class ExposureTimeCalculator() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates or retrieves the exposure time calculator yaml to create
    the asdf reference file.

    This class assumes the etc form file within the repository is used to generate the asdf file. 
    If you want to use your own form file, do: 
    rfp_etc = ExposureTimeCalculator(meta_data=my_meta, file_list=["/path/to/custom_form.yml"])
    """

    def __init__(self,
                 meta_data,
                 file_list=None,
                 outfile="roman_etc_file.asdf",
                 clobber=False
    ):
        """
        Parameters
        ----------
        meta_data: dict
            Must include a key like {"detector": "WFI03"} to identify the detector.
        file_list: list[str] | None
            When creating this reference file, a YAML form path is allowed or the form in this
            module is then used.
        outfile: str
            Output ASDF file name.
        clobber: bool
            Whether to overwrite existing ASDF file.

        Not included
        ----------
        ref_type_data: numpy array; default = None
        bit_mask: 2D integer numpy array, default = None
        """
        super().__init__(meta_data, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaETC):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaETC"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI ETC reference file."

        # Default to ETC_CONFIG if not supplied
        if file_list is None or len(file_list) == 0:
            self.form_path = ETC_FORM
            self.file_list = [str(ETC_FORM)]
        else:
            self.form_path = Path(file_list[0]).resolve()
            self.file_list = [str(self.form_path)]

        self.outfile = outfile
        self.etc_detector_form = self._get_etc_detector_form()

    def _get_etc_detector_form(self):
        """
        Load the ETC form and return a dictionary containing
        the 'common' parameters merged with the parameters
        for the detector specified in meta_data.instrument_detector.
        """

        with open(self.form_path, "r") as f:
            form = yaml.safe_load(f)

        common_form = form.get("common", {})
        detectors_form = form.get("detectors", {})

        detector_name = getattr(self.meta_data, "instrument_detector", None)
        if detector_name is None:
            raise ValueError("meta_data.instrument_detector must be set to a valid detector name (e.g. 'WFI01').")

        detector_form = detectors_form.get(detector_name)
        if detector_form is None:
            raise KeyError(f"Detector '{detector_name}' not found in ETC YAML form.")

        # Merge common and detector-specific parameters (detector takes precedence)
        merged_form = {**common_form, **detector_form}

        return merged_form
    
    # Abstract base classes not needed for ETC config reference file
    def calculate_error(self):
        return super().calculate_error()
    
    def update_data_quality_array(self):
        return super().update_data_quality_array()

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the exposure time calculator
        using the merged detector yaml form section.
        """
        #TODO replace rds.ExposureTimeRef with the correct roman_datamodels reference class when available.
        try:
            etc_datamodel_tree = rds.ExposureTimeCalcRef()
        except AttributeError:
            # use a plain dict 
            etc_datamodel_tree = {
                "meta": {},
                "form": {}
            }
        etc_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        etc_datamodel_tree["form"] = self.etc_detector_form
        return etc_datamodel_tree


# -------------------------------
# Standalone function to update form file
# -------------------------------
def update_etc_form_from_crds(output_dir="/grp/roman/RFP/DEV/scratch/etc_dump_files/"):
    """
    Update ETC YAML form with predetermined metrics for readnoise, dark current, and flat field
    values for each WFI detector from CRDS reference files.
    """
    """
    Load the ETC YAML form and set CRDS server & cache path.

    Parameters
    ----------
    output_dir : str
        Path to the directory where CRDS reference files will be cached/downloaded.
    """

    print("CRDS_SERVER_URL:", os.environ.get("CRDS_SERVER_URL"))
    print("CRDS_PATH:", os.environ.get("CRDS_PATH"))
    #TODO Test a specific context here or might have to update env var
    crds_context = crds.get_default_context()
    print(f"CRDS context: {crds_context}")

    if os.path.exists(output_dir):
        print(f"Deleting existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("Syncing CRDS reference files...")
    try:
        result = subprocess.run(
            ["crds", "sync", "--all"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running crds sync: {e.stderr}")

    if not os.path.exists(ETC_FORM):
        raise FileNotFoundError(f"ETC form file not found at {ETC_FORM}")

    with open(ETC_FORM, "r") as f:
        form = yaml.safe_load(f)

    # get a pointer to just the detectors portion of the ETC CONFIG file
    detectors_form = form.get("detectors", {})

    # -------------------------------
    # Locally download all reference files needed to update ETC config
    # -------------------------------

    readnoise_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('readnoise').reference_names()
    results = api.dump_references(crds_context, readnoise_files)
    readnoise_filepaths = list(results.values())

    dark_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('dark').reference_names()
    results = api.dump_references(crds_context, dark_files)
    dark_filepaths = list(results.values())

    # TODO figure out a way to just download one optical element for all 18 detectors using this
    # or some modification to this code
    flat_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('flat').reference_names()
    results = api.dump_references(crds_context, flat_files)
    flat_filepaths = list(results.values()) 

    saturation_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('saturation').reference_names()
    results = api.dump_references(crds_context, saturation_files)
    saturation_filepaths = list(results.values())

    # -------------------------------
    # READNOISE: update readnoise_avg with median readnoise from data array
    # -------------------------------
    for filepath in readnoise_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                val = float(np.median(ref.data))
            if det in detectors_form:
                detectors_form[det].update({
                    "readnoise_avg": val,
                    "readnoise_on": True
                })
                print(f"{det}: readnoise_avg -> {val:.2f}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # -------------------------------
    # DARK CURRENT: update dark_current_avg with mean of dark current rate array
    # -------------------------------
    for filepath in dark_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                val = float(np.mean(ref.dark_slope))
            if det in detectors_form:
                detectors_form[det].update({
                    "dark_current_avg": val,
                    "dark_current_on": True
                })
                print(f"{det}: dark_current_avg -> {val:.3f}")
        except Exception as e:
            print(f"Failed to process dark current {filepath}: {e}")

    # -------------------------------
    # FLAT FIELD: update ff_electrons
    # -------------------------------
    for filepath in flat_filepaths:
        try:
            with rdm.open(filepath) as ref:
                if ref.meta.instrument.optical_element == 'F062':
                    det = ref.meta.instrument.detector
                    std_val = float(np.std(ref.data))
                    ff_electrons = 1.0 / (std_val ** 2)
                    if det in detectors_form:
                        detectors_form[det].update({
                            "ff_electrons": ff_electrons,
                            "ffnoise": True
                        })
                        print(f"{det}: ff_electrons -> {ff_electrons:.2f}")
        except Exception as e:
            print(f"Failed to process flat field {filepath}: {e}")

    # -------------------------------
    # SATURATION: update saturation_fullwell
    # -------------------------------
    for filepath in saturation_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                val = float(np.amax(ref.data))
            if det in detectors_form:
                detectors_form[det].update({
                    "saturation_fullwell": val,
                    "saturation_on": True
                })
                print(f"{det}: saturation_fullwell -> {val:.1f}")
        except Exception as e:
            print(f"Failed to process saturation {filepath}: {e}")

    # -------------------------------
    # Write updated config
    # -------------------------------
    with open(ETC_FORM, "w") as f:
        yaml.safe_dump(form, f, sort_keys=False)

    print(f"Updated ETC form file saved to: {ETC_FORM}")
