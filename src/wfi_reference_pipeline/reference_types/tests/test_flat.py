import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wfi_reference_pipeline.constants import REF_TYPE_DARK, REF_TYPE_FLAT
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Flat class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_FLAT)
    return test_meta.meta_flat


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid ref_type_data array (flat field image)."""
    return np.ones((4088, 4088)).astype(np.float32)  # Simulate a valid flat field image


@pytest.fixture
def valid_ref_type_data_cube():
    """Fixture for generating valid ref_type_data cube (flat field cube)."""
    return np.ones((5, 4088, 4088)).astype(np.float32)  # Simulate a valid flat field data cube


@pytest.fixture
def flat_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a Flat object with a valid data array."""
    flat_object_with_data_array = Flat(meta_data=valid_meta_data,
                                       ref_type_data=valid_ref_type_data_array)
    yield flat_object_with_data_array


@pytest.fixture
def flat_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a Flat object with a valid data cube."""
    flat_object_with_data_cube = Flat(meta_data=valid_meta_data,
                                      ref_type_data=valid_ref_type_data_cube)
    yield flat_object_with_data_cube


class TestFlat:

    def test_flat_instantiation_with_valid_ref_type_data_array(self, flat_object_with_data_array):
        """
        Test that Flat object is created successfully with valid input data array.
        """
        assert isinstance(flat_object_with_data_array, Flat)
        assert flat_object_with_data_array.flat_image.shape == (4088, 4088)

    def test_flat_instantiation_with_valid_ref_type_data_cube(self, flat_object_with_data_cube):
        """
        Test that Flat object is created successfully with valid input data cube.
        """
        assert isinstance(flat_object_with_data_cube, Flat)
        assert flat_object_with_data_cube.data_cube is not None
        # Ensure image is not created yet
        assert flat_object_with_data_cube.flat_image is None

    def test_flat_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that Flat raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
        with pytest.raises(TypeError):
            Flat(meta_data=bad_test_meta.meta_dark,
                 ref_type_data=valid_ref_type_data_array)

    def test_flat_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Flat raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Flat(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')

    @patch("asdf.open")
    def test_flat_instantiation_with_file_list(self, mock_asdf_open, valid_meta_data):
        """
        Test that Flat object handles file list input correctly.
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "roman": {
                "data": np.zeros((10, 10))  # Mocking data
            }
        }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits", "file2.fits"]
        flat_obj = Flat(meta_data=valid_meta_data, file_list=mock_file_list)

        assert flat_obj.num_files == 2

    def test_make_flat_image_with_data_cube(self, valid_meta_data, valid_ref_type_data_cube):
        """
        Test that the make_mask_image method successfully creates the flat image.
        """
        flat_object = Flat(meta_data=valid_meta_data,
                           ref_type_data=valid_ref_type_data_cube)
        flat_object.make_flat_image()
        assert flat_object.flat_image is not None

    def test_calculate_error_from_flat_with_data_cube(self, flat_object_with_data_cube):
        """
        Test calculate_error from input data cube.
        """
        flat_object_with_data_cube.calculate_error(fill_random=True)
        assert flat_object_with_data_cube.flat_error is not None
        assert flat_object_with_data_cube.flat_error.shape == (4088, 4088)

    @skip_on_github
    def test_calculate_error_with_individual_images(self, flat_object_with_data_cube):
        """
        Test calculate_error with a user-supplied error array.
        """
        custom_individual = np.ones((10, 4088, 4088), dtype=np.float32)
        flat_object_with_data_cube.calculate_error(ind_flat_array=custom_individual,
                                                   nsamples=5, nboot=10)
        assert np.array_equal(
            flat_object_with_data_cube.flat_error, np.zeros((4088, 4088), dtype=np.float32))

    def test_update_data_quality_array(self, flat_object_with_data_array):
        """
        Test update_data_quality_array adds DQ flags for low QE pixels.
        """
        low_qe_threshold = 0.2
        # Simulate a low-QE pixel
        flat_object_with_data_array.flat_image[2000, 2000] = 0.1
        flat_object_with_data_array.update_data_quality_array(
            low_qe_threshold=low_qe_threshold)
        assert flat_object_with_data_array.dq_mask[2000, 2000] > 0

    def test_populate_datamodel_tree(self, flat_object_with_data_array):
        """
        Test that the data model tree is correctly populated in the Flat object.
        """
        data_model_tree = flat_object_with_data_array.populate_datamodel_tree()

        # Assuming the Flat data model includes:
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree
        assert 'dq' in data_model_tree
        assert 'err' in data_model_tree

        # Check the shape and dtype of the 'data' array
        assert data_model_tree['data'].shape == (4088, 4088)
        assert data_model_tree['data'].dtype == np.float32
        # Check the shape and dtype of the 'dq' array
        assert data_model_tree['dq'].shape == (4088, 4088)
        assert data_model_tree['dq'].dtype == np.uint32
        # Check the shape and dtype of the 'err' array
        assert data_model_tree['err'].shape == (4088, 4088)
        assert data_model_tree['err'].dtype == np.float32

    def test_flat_outfile_default(self, flat_object_with_data_array):
        """
        Test that the default outfile name is correct in the Flat object with the assumption
        that the default name is 'roman_flat.asdf'
        """
        assert flat_object_with_data_array.outfile == "roman_flat.asdf"
