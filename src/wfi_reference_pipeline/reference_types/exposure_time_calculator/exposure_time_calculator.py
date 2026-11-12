
import os
from pathlib import Path

import numpy as np
import roman_datamodels as rdm
import yaml
from roman_datamodels.datamodels import EtcRefModel

from wfi_reference_pipeline.constants import (
    REF_TYPE_DARK,
    REF_TYPE_FLAT,
    REF_TYPE_GAIN,
    REF_TYPE_READNOISE,
    REF_TYPE_SATURATION,
)

from wfi_reference_pipeline.reference_types.gain.gain import (
    GAIN_CORRECTION_FACTOR,
)
from wfi_reference_pipeline.resources.wfi_meta_exposure_time_calculator import (
    WFIMetaETC,
)

from wfi_reference_pipeline.utilities.crds_handler import CrdsHandler

from ..reference_type import ReferenceType

ETC_FORM = (Path(__file__).parent / "exposure_time_calculator_form.yml").resolve()

# Optical element whose CRDS flats are used for the flat field values in the ETC form.
ETC_FLAT_OPTICAL_ELEMENT = "F062"

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

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the exposure time calculator
        using the merged detector yaml form section.
        """

        etc_datamodel_tree = EtcRefModel()
        etc_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        etc_datamodel_tree['form'] = self.etc_detector_form

        return etc_datamodel_tree

    # Abstract base classes not needed for ETC config reference file
    def calculate_error(self):
        return super().calculate_error()

    def update_data_quality_array(self):
        return super().update_data_quality_array()

# -------------------------------
# Standalone function to update form file
# -------------------------------
def update_etc_form_from_crds():
    """
    Update ETC YAML form with predetermined metrics for readnoise, dark current, and flat field
    values for each WFI detector from CRDS reference files.
    
    Reference files are retrieved with CrdsHandler from the latest CRDS context: for each
    reftype, the file that applies now for each of the 18 detectors. Files are cached in CRDS_PATH.
    """

    if not os.path.exists(ETC_FORM):
        raise FileNotFoundError(f"ETC form file not found at {ETC_FORM}")

    with open(ETC_FORM, "r") as f:
        form = yaml.safe_load(f)

    # get a pointer to just the detectors portion of the ETC CONFIG file
    detectors_form = form.get("detectors", {})

    # -------------------------------
    # Locally download all reference files needed to update ETC config, one per detector
    # -------------------------------
    crds_handler = CrdsHandler()

    # ETC operates in e- and e-/s
    # All detector parameters need to be gain-corrected
    gain_filepaths = list(crds_handler.get_reference_files(REF_TYPE_GAIN).values())
    readnoise_filepaths = list(crds_handler.get_reference_files(REF_TYPE_READNOISE).values())
    dark_filepaths = list(crds_handler.get_reference_files(REF_TYPE_DARK).values())
    flat_filepaths = list(
        crds_handler.get_reference_files(
            REF_TYPE_FLAT, optical_element=ETC_FLAT_OPTICAL_ELEMENT
        ).values()
    )
    saturation_filepaths = list(crds_handler.get_reference_files(REF_TYPE_SATURATION).values())


    # -------------------------------
    # GAIN: grab gain files and take the median of the data array
    # The gain files from CRDS do not have the correction factor to account for IPC baked in
    # This factor needs to be applied across-the-board to convert DNs to electrons
    # See the Gain module for the actual value and its full explanation
    # -------------------------------
    gain_vals = {}          # Create an empty dictionary to store the gain values for each detector
    for filepath in gain_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                val = float(np.median(ref.data))

                # Correct the gain values to account for IPC + small uncorrected nonlinearity
                # Value is from T. Brandt
                # See the Gain module for the actual value and its full explanation
                val_corrected = val / GAIN_CORRECTION_FACTOR
                print(f"{det}: gain -> {val:.2f}, gain_corrected -> {val_corrected:.2f}")

                # Save the gain values in the dictionary to convert DNs to electrons 
                # for the rest of the detector parameters
                gain_vals[det] = val_corrected
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # -------------------------------
    # READNOISE: update readnoise with median readnoise from data array
    # -------------------------------
    for filepath in readnoise_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                val = float(np.median(ref.data)) * gain_vals[det]   # ETC operates in electrons - applying the gain to convert DN to electron
            if det in detectors_form:
                detectors_form[det].update({
                    "readnoise": round(val, 3),     # ETC's numerical precision is 2 or 3 decimal places for the pixel-scale related values
                    "readnoise_on": True
                })
                print(f"{det}: readnoise -> {val:.3f}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


    # -------------------------------
    # DARK CURRENT: update dark_current with median of dark current rate array
    # As of 06/09/26, we are hard-coding the dark value to be 0.01 until 
    # new reference files and our workflow are further refined
    # -------------------------------
    for filepath in dark_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                #val = float(np.median(ref.dark_slope)) * gain_vals[det]
                val = 0.01      # Hard-coding to a value as of 06/08/2026 until we converge on the workflow
            if det in detectors_form:
                detectors_form[det].update({
                    "dark_current": round(val, 3),  # ETC's numerical precision is 2 or 3 decimal places for the pixel-scale related values
                    "dark_current_on": True
                })
                print(f"{det}: dark_current -> {val:.3f}")
        except Exception as e:
            print(f"Failed to process dark current {filepath}: {e}")

    # -------------------------------
    # FLAT FIELD: update flat_field_electrons
    # As of R2026.1, the ETC is set to compute the flat fielding error from the total number of electrons in the superflat image.
    # Starting with R2027.3, RTB requested that the engine takes the flat field uncertainty we provide in the flat reference file.
    # When the time comes, rename flat_field_electrons in the yaml to flat_field_uncertainty
    # -------------------------------

    for filepath in flat_filepaths:
        try:
            with rdm.open(filepath) as ref:
                if ref.meta.instrument.optical_element == ETC_FLAT_OPTICAL_ELEMENT:
                    det = ref.meta.instrument.detector
                    ff_electrons = 1000000      # Hard-coding to a nominal value as of 06/08/2026
                    '''
                    # Replace the above hard-coded value with this part
                    # In the future, the engine will be using the flat field uncertainty provided by RTB as-is
                    # rather than computing the value from the number of electrons from the superflat itself.
                    
                    if (ref.err != None) or (ref.err != 0):
                        ff_electrons = float(np.nanmedian(ref.err)) * gain_vals[det]
                    else:
                        ff_electrons = np.nanstd(ref.data) * gain_vals[det]
                    '''
                    if det in detectors_form:
                        detectors_form[det].update({
                            "flat_field_electrons": ff_electrons,   # Rename this parameter to flat_field_uncertainty
                            "flat_field_noise_on": True
                        })
                        print(f"{det}: flat_field_electrons -> {ff_electrons}")
        except Exception as e:
            print(f"Failed to process flat field {filepath}: {e}")

    # -------------------------------
    # SATURATION: update saturation_fullwell
    # As of 05/29/2026, set the fullwell depth to a typical value (100,000 electrons)
    # This part of the script will be re-visited in the future(after commissioning) to re-evaluate the value
    # -------------------------------
    for filepath in saturation_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                # val = float(np.amax(ref.data)) * gain_vals[det]
                val = 100000       # Hard-coding to a typical value as of 05/29/2026
            if det in detectors_form:
                detectors_form[det].update({
                    "saturation_fullwell": val,
                    "saturation_on": True
                })
                print(f"{det}: saturation_fullwell -> {val}")
        except Exception as e:
            print(f"Failed to process saturation {filepath}: {e}")


    # -------------------------------
    # Write updated config
    # -------------------------------
    with open(ETC_FORM, "w") as f:
        yaml.safe_dump(form, f, sort_keys=False)

    print(f"Updated ETC form file saved to: {ETC_FORM}")


# -------------------------------
# Optional: for RTB internal use only
# Extra function to add the following comments only to WFI01 to have information of what the detector keys are for
# This information is internal to RTB and does not appear in the converted ASDF file
# -------------------------------
def add_eol_comments_for_wfi01(yaml_file, comment_map):
    """
    Update ETC YAML form once more to add useful information on what each parameter is in the ETC
    This function is useful for the RTB internal usage. 
    """
    """

    Parameters
    ----------
    yaml_file : str
        Path to the directory where the yaml file leaves.
    comment_map: dict
        A dictionary containing the key and the comments 
    """
    with open(yaml_file, 'r') as f:
        lines = f.readlines()

    modified_lines = []

    # A counter is needed to only add comments to the WFI01 contents
    count = 0
    
    for line in lines:
        count += 1

        # WFI01's parameters are from lines 23-31
        if (count >= 22) and (count <= 31):
            # Check if the line contains a key we want to comment
            for key, comment in comment_map.items():
                # Matches keys followed by a colon
                if (key+':') in line:
                    # Strip trailing newlines, spaces, append comment, and add newline back
                    stripped = line.rstrip('\r\n')
                    line = f"{stripped}  # {comment}\n"                    
        else:
            # Do not touch anything out of WFI01's content
            line = line
        modified_lines.append(line)

    with open(yaml_file, 'w') as f:
        f.writelines(modified_lines)


