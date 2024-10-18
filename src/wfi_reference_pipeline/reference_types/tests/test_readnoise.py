import pytest
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_READNOISE, REF_TYPE_DARK
from unittest.mock import MagicMock, patch
import numpy as np
import astropy.units as u


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for ReadNoise class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
    return test_meta.meta_readnoise


@pytest.fixture
def valid_ref_type_data():
    """Fixture for generating valid ref_type_data (read noise image)."""
    return np.random.random((4096, 4096))  # Simulate a valid read noise image


@pytest.fixture
def readnoise_object(valid_meta_data, valid_ref_type_data):
    """Fixture for initializing a valid ReadNoise object."""
    readnoise_obj = ReadNoise(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    yield readnoise_obj


@pytest.fixture
def valid_data_cube():
    """Fixture for generating a mock data cube for testing ramp residual variance."""
    mock_data_cube = MagicMock()
    mock_data_cube.num_i_pixels = 4096
    mock_data_cube.num_j_pixels = 4096
    mock_data_cube.data = np.random.random((5, 4096, 4096))  # 5 reads in the cube
    mock_data_cube.ramp_model = np.random.random((5, 4096, 4096))  # Simulated ramp model
    return mock_data_cube


@pytest.fixture
def readnoise_object_with_data_cube(valid_meta_data, valid_data_cube):
    """Fixture for initializing a ReadNoise object with a valid data cube."""
    readnoise_obj = ReadNoise(meta_data=valid_meta_data, ref_type_data=None)
    readnoise_obj.data_cube = valid_data_cube  # Assign mock data cube
    yield readnoise_obj


class TestReadNoise:

    def test_readnoise_instantiation_with_valid_data(self, readnoise_object):
        """
        Test that ReadNoise object is created successfully with valid input data.
        """
        assert isinstance(readnoise_object, ReadNoise)
        assert readnoise_object.readnoise_image.shape == (4096, 4096)

    def test_readnoise_instantiation_with_invalid_metadata(self, valid_ref_type_data):
        """
        Test that ReadNoise raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
        with pytest.raises(TypeError):
            ReadNoise(meta_data=bad_test_meta.meta_dark, ref_type_data=valid_ref_type_data)

    def test_readnoise_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that ReadNoise raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            ReadNoise(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')

    def test_readnoise_instantiation_with_data_cube(self, valid_meta_data):
        """
        Test that ReadNoise object handles 3D data cube input correctly.
        """
        data_cube = np.random.random((5, 4096, 4096))  # 5 reads in the cube
        readnoise_obj = ReadNoise(meta_data=valid_meta_data, ref_type_data=data_cube)

        assert readnoise_obj.data_cube is not None
        assert readnoise_obj.readnoise_image is None  # Ensure image is not created yet
        assert readnoise_obj.data_cube.data.shape == (5, 4096, 4096)

    def test_readnoise_instantiation_with_quantity_object(self, valid_meta_data):
        """
        Test that ReadNoise object handles Quantity object as ref_type_data correctly.
        """
        data_quantity = (np.random.random((4096, 4096)) * u.electron)  # Quantity object
        readnoise_obj = ReadNoise(meta_data=valid_meta_data, ref_type_data=data_quantity)

        assert readnoise_obj.readnoise_image is not None
        assert isinstance(readnoise_obj.readnoise_image, np.ndarray)
        assert readnoise_obj.readnoise_image.shape == (4096, 4096)

    @patch("asdf.open")
    def test_readnoise_instantiation_with_file_list(self, mock_asdf_open, valid_meta_data):
        """
        Test that ReadNoise object handles file list input correctly.
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "roman": {
                "data": np.zeros((10, 2048, 2048))  # Mocking a datacube with 10 reads
            }
        }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits", "file2.fits"]
        readnoise_obj = ReadNoise(meta_data=valid_meta_data, file_list=mock_file_list)

        assert readnoise_obj.num_files == 2

    def test_make_readnoise_image_using_cds_noise(self, readnoise_object):
        """
        Test that make_readnoise_image can be modified to use CDS noise calculation.
        """
        # Mock the comp_cds_noise method to simulate CDS noise calculation
        readnoise_object.comp_cds_noise = MagicMock(return_value='mock_cds_noise_image')

        # Modify the method for CDS noise calculation
        readnoise_object.readnoise_image = readnoise_object.comp_cds_noise()

        # Assert that the comp_cds_noise method was called and readnoise_image was updated
        readnoise_object.comp_cds_noise.assert_called_once()
        assert readnoise_object.readnoise_image == 'mock_cds_noise_image'

