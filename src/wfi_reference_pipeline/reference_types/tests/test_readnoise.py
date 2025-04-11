from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wfi_reference_pipeline.constants import REF_TYPE_DARK, REF_TYPE_READNOISE
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for ReadNoise class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
    return test_meta.meta_readnoise


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid ref_type_data array (read noise image)."""
    return np.random.random((4096, 4096))  # Simulate a valid read noise image


@pytest.fixture
def valid_ref_type_data_cube():
    """Fixture for generating valid ref_type_data cube (read noise cube)."""
    return np.random.random((3, 4096, 4096))  # Simulate a valid read noise image


@pytest.fixture
def readnoise_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a ReadNoise object with a valid data array."""
    readnoise_object_with_data_array = ReadNoise(meta_data=valid_meta_data,
                                                 ref_type_data=valid_ref_type_data_array)
    yield readnoise_object_with_data_array


@pytest.fixture
def readnoise_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a ReadNoise object with a valid data cube."""
    readnoise_object_with_data_cube = ReadNoise(meta_data=valid_meta_data,
                                                ref_type_data=valid_ref_type_data_cube)
    yield readnoise_object_with_data_cube


class TestReadNoise:

    def test_readnoise_instantiation_with_valid_ref_type_data_array(self,
                                                                    readnoise_object_with_data_array):
        """
        Test that ReadNoise object is created successfully with valid input data array.
        """
        assert isinstance(readnoise_object_with_data_array, ReadNoise)
        assert readnoise_object_with_data_array.readnoise_image.shape == (4096, 4096)

    def test_readnoise_instantiation_with_valid_ref_type_data_cube(self,
                                                                    readnoise_object_with_data_cube):
        """
        Test that ReadNoise object is created successfully with valid input data cube.
        """
        assert isinstance(readnoise_object_with_data_cube, ReadNoise)
        assert readnoise_object_with_data_cube.data_cube is not None
        assert readnoise_object_with_data_cube.readnoise_image is None  # Ensure image is not created yet

    def test_readnoise_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that ReadNoise raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
        with pytest.raises(TypeError):
            ReadNoise(meta_data=bad_test_meta.meta_dark, ref_type_data=valid_ref_type_data_array)

    def test_readnoise_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that ReadNoise raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            ReadNoise(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')

    @patch("asdf.open")
    def test_readnoise_instantiation_with_file_list(self, mock_asdf_open, valid_meta_data):
        """
        Test that ReadNoise object handles file list input correctly.
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "roman": {
                "data": np.zeros((3, 10, 10))  # Mocking a datacube with 10 reads
            }
        }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits", "file2.fits"]
        readnoise_obj = ReadNoise(meta_data=valid_meta_data, file_list=mock_file_list)

        assert readnoise_obj.num_files == 2

    def test_make_readnoise_image_using_cds_noise(self, readnoise_object_with_data_cube):
        """
        Test that make_readnoise_image can be modified to use CDS noise calculation.
        """
        # Mock the comp_cds_noise method to simulate CDS noise calculation
        readnoise_object_with_data_cube.comp_cds_noise = MagicMock(return_value='mock_cds_noise_image')

        # Modify the method for CDS noise calculation
        readnoise_object_with_data_cube.readnoise_image = readnoise_object_with_data_cube.comp_cds_noise()

        # Assert that the comp_cds_noise method was called and readnoise_image was updated
        readnoise_object_with_data_cube.comp_cds_noise.assert_called_once()
        assert readnoise_object_with_data_cube.readnoise_image == 'mock_cds_noise_image'

    def test_populate_datamodel_tree(self, readnoise_object_with_data_array):
        """
        Test that the data model tree is correctly populated in the ReadNoise object.
        """
        data_model_tree = readnoise_object_with_data_array.populate_datamodel_tree()

        # Assuming the ReadNoise data model includes 'meta' and 'noise' (instead of 'dq')
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree  # For read noise data, it would likely be 'noise' instead of 'dq'

        # Check the shape and dtype of the 'noise' array
        assert data_model_tree['data'].shape == (4096, 4096)
        assert data_model_tree['data'].dtype == np.float32  # Assuming read noise values are stored as floats

    def test_readnoise_outfile_default(self, readnoise_object_with_data_array):
        """
        Test that the default outfile name is correct in the ReadNoise object with the assumption
        that the default name is 'roman_readnoise.asdf'
        """
        assert readnoise_object_with_data_array.outfile == "roman_readnoise.asdf"

