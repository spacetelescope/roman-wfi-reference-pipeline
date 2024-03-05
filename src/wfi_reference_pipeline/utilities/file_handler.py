from pathlib import Path

def _get_prepped_file_suffix(ref_type):
    return ref_type.upper() + '_PREPPED.asdf'


def remove_existing_prepped_files_for_ref_type(prep_dir, ref_type):
    # TODO create check_ref_type
    #if valid_ref_type(ref_Type):
    file_path_to_remove = Path(prep_dir)
    print(file_path_to_remove.name)
    matching_files = file_path_to_remove.glob('*' + _get_prepped_file_suffix(ref_type))

    # Delete each matching file
    for file in matching_files:
        file.unlink()
        print(f"removing {file}")

def format_prep_output_file_path(prep_dir, filename, ref_type):
    """Return a file path for a prepped file using established formatting
        /PATH/IN/CONFIG/filenameREFTYPE_PREPPED.asdf

    Returns
    -------
    settings : string
        Path for prepped file
    """
    prepped_filename = filename + _get_prepped_file_suffix(ref_type)
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