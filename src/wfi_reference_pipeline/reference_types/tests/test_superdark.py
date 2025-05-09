import os
from pathlib import Path

import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DARK_LONG_NUM_READS,
    DARK_SHORT_NUM_READS,
)
from wfi_reference_pipeline.reference_types.dark.superdark import SuperDark


# Temporary subclass to test the abstract SuperDark class
class TmpSuperDark(SuperDark):
    def generate_superdark(self):
        # Mock implementation: simply fill superdark with fake data for testing
        self.superdark = self._generate_test_superdark_cube()

    def _generate_test_superdark_cube(self):
        # Return a 3D array with shape (num_reads, 4096, 4096)
        import numpy as np
        return np.ones((self.short_dark_num_reads, 4096, 4096))


@pytest.fixture(scope="class")
def superdark_object(tmp_path_factory):
    """
    Create a TmpSuperDark instance for testing.
    """
    input_path = tmp_path_factory.mktemp("data")

    short_dark_files = [os.path.join(input_path, f'short_dark_{i}.fits') for i in range(3)]
    long_dark_files = [os.path.join(input_path, f'long_dark_{i}.fits') for i in range(2)]

    # Write dummy files
    for file in short_dark_files + long_dark_files:
        Path(file).touch()

    obj = TmpSuperDark(short_dark_files,
                       long_dark_files,
                       DARK_SHORT_NUM_READS,
                       DARK_LONG_NUM_READS,
                       "WFI01"
                       )
    obj.generate_superdark()
    yield obj


class TestSuperDark:

    def test_superdark_initialization(self):
        """
        Test if the SuperDark object initializes properly with valid input.
        """
        short_dark_files = ["/mock/input/path/file_WFI01_short.asdf", "/mock/input/path/file_WFI01_short2.asdf"]
        long_dark_files = ["/mock/input/path/file_WFI01_long.asdf"]

        obj = TmpSuperDark(short_dark_files,
                           long_dark_files,
                           DARK_SHORT_NUM_READS,
                           DARK_LONG_NUM_READS,
                           "WFI01"
                           )

        assert obj.wfi_detector_str == "WFI01"
        assert obj.outfile == "WFI01_superdark.asdf"
        assert len(obj.file_list) == 3
        assert obj.meta_data['detector'] == "WFI01"

    def test_superdark_invalid_detector(self):
        """
        Test for invalid WFI detector string initialization.
        """

        short_dark_files = ["/mock/input/path/file_WFI99_short.fits"]  # Invalid detector

        with pytest.raises(ValueError, match="Invalid WFI detector ID WFI99; Must be WFI01-WFI18"):
            TmpSuperDark(short_dark_files,
                         short_dark_files,
                         DARK_SHORT_NUM_READS,
                         DARK_LONG_NUM_READS,
                         "WFI99"
                         )

    def test_superdark_mismatched_short_long_detectors(self):
        """
        Test for mismatched detectors in short/long dark file lists.
        """
        short_dark_files = ["/mock/input/path/file_WFI01_short.asdf", "/mock/input/path/file_WFI01_short2.asdf"]
        long_dark_files = ["/mock/input/path/file_WFI02_long.asdf"]

        with pytest.raises(ValueError, match="More than one WFI detector ID found"):
            TmpSuperDark(short_dark_files,
                         long_dark_files,
                         DARK_SHORT_NUM_READS,
                         DARK_LONG_NUM_READS,
                         "WFI01"
                         )

    def test_superdark_mismatched_short_detectors(self):
        """
        Test for mismatched detectors in short/long dark file lists.
        """
        short_dark_files = ["/mock/input/path/file_WFI01_short.asdf", "/mock/input/path/file_WFI02_short2.asdf"]
        long_dark_files = ["/mock/input/path/file_WFI02_long.asdf"]
        with pytest.raises(ValueError, match="More than one WFI detector ID found"):
            TmpSuperDark(short_dark_files,
                         long_dark_files,
                         DARK_SHORT_NUM_READS,
                         DARK_LONG_NUM_READS,
                         "WFI01"
                         )

    def test_generate_outfile_permissions(self, tmp_path):
        """
        Test if the outfile is generated with correct permissions.
        """
        # Setup the SuperDark object
        short_dark_files = ["/mock/input/path/file_WFI01_short.fits"]
        long_dark_files = ["/mock/input/path/file_WFI01_long.fits"]

        obj = TmpSuperDark(short_dark_files,
                           long_dark_files,
                           DARK_SHORT_NUM_READS,
                           DARK_LONG_NUM_READS,
                           "WFI01"
                           )

        # Mock the superdark data for the test
        obj.superdark = np.zeros((1, 4096, 4096))

        # Generate the outfile in the temporary path
        obj.outfile = tmp_path / "test_superdark.asdf"
        obj.generate_outfile(file_permission=0o644)

        try:
            assert obj.outfile.exists()
            assert oct(obj.outfile.stat().st_mode)[-3:] == "644"
        finally:
            # Cleanup the generated file
            if obj.outfile.exists():
                obj.outfile.unlink()

    @pytest.mark.skip(reason="Temporarily disabled test")
    def test_superdark_borders_set_to_zero(self, superdark_object, tmp_path):
        """
        Test if the borders of the superdark cube are correctly set to zero after calling generate_outfile().
        """
        # Make sure superdark is all ones initially
        assert np.all(superdark_object.superdark == 1)
        superdark_object.outfile = tmp_path / "test_superdark.asdf"
        # Call generate_outfile, which should zero out the borders
        superdark_object.generate_outfile()

        # Check the borders (the first and last 4 rows/columns) are set to zero
        superdark = superdark_object.superdark

        # Check the borders in the first axis (rows and columns)
        assert np.all(superdark[:, :4, :] == 0)  # first 4 rows in all images
        assert np.all(superdark[:, -4:, :] == 0)  # last 4 rows in all images
        assert np.all(superdark[:, :, :4] == 0)  # first 4 columns in all images
        assert np.all(superdark[:, :, -4:] == 0)  # last 4 columns in all images

        # Ensure non-border pixels are still ones
        assert np.all(superdark[:, 4:-4, 4:-4] == 1)

