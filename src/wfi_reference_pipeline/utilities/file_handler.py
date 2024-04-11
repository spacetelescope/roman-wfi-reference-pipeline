import logging
from pathlib import Path

from wfi_reference_pipeline import constants


class FileHandler:
    """
    Utility class to safely handle reftype specific pipeline file interactions
    This class is designed to ONLY be initialized in the Pipeline base class
    All information is supplied on initialization so multiple pipeline files do not interact
    This class should be initialized as part of a base pipeline initialization procedure AFTER config values have been read and stored
    """

    def __init__(self, ref_type, prep_path, pipeline_out_path):
        if ref_type not in constants.WFI_REF_TYPES:
            raise ValueError(
                f"ref_type {ref_type} not valid, must be: {constants.WFI_REF_TYPES}"
            )

        self.ref_type = ref_type
        self.prep_path = Path(prep_path)
        self.pipeline_out_path = Path(pipeline_out_path)

    def _get_prepped_file_suffix(self):
        """All prepped files should have the same suffix of <ref_type>_PREPPED.asdf"""
        return self.ref_type + "_PREPPED.asdf"

    def remove_existing_prepped_files_for_ref_type(self):
        """Remove previous PREPPED files for the reference type
        There should only ever be the most recent run of prepped files foreach ref_type
        """
        # if valid_ref_type(ref_Type):
        file_path_to_remove = self.prep_path
        print(file_path_to_remove.name)
        matching_files = file_path_to_remove.glob("*" + self._get_prepped_file_suffix())

        # Delete each matching file
        for file in matching_files:
            file.unlink()
            logging.info(f"Cleaning Prep folder for {self.ref_type}: Removing {file}")

    def format_prep_output_file_path(self, filename):
        """Return a file path for a prepped file using established formatting
            /PATH/IN/CONFIG/filenameREFTYPE_PREPPED.asdf

        Returns
        -------
        output_path: Path()
            Path for prepped file
        """
        prepped_filename = filename + self._get_prepped_file_suffix()
        output_path = self.prep_path / prepped_filename
        return output_path

    def format_pipeline_output_file_path(self, filename):
        """Return a file path for a calibrated file using established formatting
            /PATH/IN/CONFIG/filename

        Returns
        -------
        output_path : Path()
            Path for output file
        """
        # TODO add identifiers to filename (date? other format?), and asdf extension if it doesn't exist
        # TODO delete existing files like we do with prep???
        output_path = self.pipeline_out_path / filename
        return output_path
