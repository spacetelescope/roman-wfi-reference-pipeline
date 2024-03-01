from pathlib import Path

from wfi_reference_pipeline.utilities.config_handler import get_datafiles_config

def format_prep_output_file_path(prep_dir, filename, ref_type):
    """Return a file path for a prepped file using established formatting
        /PATH/IN/CONFIG/filenameREFTYPE_PREPPED.asdf

    Returns
    -------
    settings : string
        Path for prepped file
    """
    prepped_filename = filename + ref_type.upper() + '_PREPPED.asdf'
    output_path = Path(prep_dir) / prepped_filename
    return output_path

def format_calibrated_output_file_path(out_dir, filename):
    """Return a file path for a calibrated file using established formatting
        /PATH/IN/CONFIG/filename

    Returns
    -------
    settings : string
        Path for prepped file
    """
    # TODO add identifiers to filename (date? other format?), and asdf extension if it doesn't exist
    output_path = Path(out_dir) / filename
    return output_path