from pathlib import Path

from wfi_reference_pipeline.utilities.config_handler import get_datafiles_config

def get_prep_output_file_path(self, filename, ref_type):
    """Return a file path for a prepped file using established formatting
        /PATH/IN/CONFIG/filenameREFTYPE_PREPPED.asdf

    Returns
    -------
    settings : string
        Path for prepped file
    """
    prep_dir = get_datafiles_config()["prep_dir"]
    prepped_filename = filename + ref_type.upper() + '_PREPPED.asdf'
    output_path = Path(prep_dir) / prepped_filename
    return output_path

def get_calibrated_output_file_path(self, filename):
    """Return a file path for a calibrated file using established formatting
        /PATH/IN/CONFIG/filename

    Returns
    -------
    settings : string
        Path for prepped file
    """
    out_dir = get_datafiles_config()["calibrated_dir"]
    output_path = Path(out_dir) / filename
    return output_path