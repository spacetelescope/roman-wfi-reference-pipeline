import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from romancal.lib import dqflags

from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT, REF_TYPE_DARK, REF_TYPE_READNOISE
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Dark class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
    return test_meta.meta_dark


@pytest.fixture
def valid_ref_type_data_cube():
    """Fixture for generating valid ref_type_data (read cube)."""
    valid_ref_type_data_cube, _ = simulate_dark_reads(5)  # Simulate a 5-read data cube
    return valid_ref_type_data_cube


@pytest.fixture
def dark_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a Flat object with a valid data cube."""
    dark_object_with_data_cube = Dark(meta_data=valid_meta_data,
                                      ref_type_data=valid_ref_type_data_cube)
    yield dark_object_with_data_cube


@pytest.fixture
def dark_rate_image_3_by_3():
    """Fixture for a testable dark rate image.

    Returning array in top row that is above threshold values,
    middle row which is equal to threshold values, and bottom
    row that is below.

    array flags = [ hot, warm, good,
                    hot, warm, dead,
                    good, dead, dead]
    """
    return np.array([
        [2.1, 1.1, 0.2],  # should return hot, warm, no flag set
        [2.0, 1.0, 0.1],  # should return hot, warm, dead
        [0.5, 0.0, -0.1],  # should return no flag set, dead, dead
    ])


class TestDark:

    def test_dark_instantiation_with_valid_ref_type_data_cube(self, dark_object_with_data_cube):
        """
        Test that Dark object is created successfully with valid input data cube.
        """
        assert isinstance(dark_object_with_data_cube, Dark)
        assert dark_object_with_data_cube.data_cube is not None
        assert dark_object_with_data_cube.dark_rate_image is None  # Ensure image is not created yet
        assert dark_object_with_data_cube.dark_rate_image_error is None  # Ensure error array is not created yet

    def test_dark_instantiation_with_invalid_metadata(self, dark_object_with_data_cube):
        """
        Test that Flat raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Dark(meta_data=bad_test_meta.meta_readnoise, ref_type_data=dark_object_with_data_cube)

    def test_dark_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Flat raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Dark(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')

    def test_make_rate_image_from_data_cube_default_fit_order(self, dark_object_with_data_cube):
        """
        Test that the method make_rate_image_from_data_cube works with default fit_order.
        """
        mock_return_image = np.random.rand(3, 3)  # Create a dummy dark rate image
        mock_return_error_image = np.random.rand(3, 3)  # Dummy error image
        dark_object_with_data_cube.data_cube.fit_cube = MagicMock()
        dark_object_with_data_cube.data_cube.rate_image = mock_return_image
        dark_object_with_data_cube.data_cube.rate_image_err = mock_return_error_image

        dark_object_with_data_cube.make_rate_image_from_data_cube()

        # Assert that fit_cube was called with degree=1 (default fit_order)
        dark_object_with_data_cube.data_cube.fit_cube.assert_called_once_with(degree=1)

        # Check that dark_rate_image and dark_rate_image_error are set
        assert dark_object_with_data_cube.dark_rate_image is not None
        assert dark_object_with_data_cube.dark_rate_image_error is not None

        # Additional checks
        assert dark_object_with_data_cube.dark_rate_image.shape == mock_return_image.shape
        assert dark_object_with_data_cube.dark_rate_image_error.shape == mock_return_error_image.shape

    def test_make_rate_image_from_data_cube_custom_fit_order(self, dark_object_with_data_cube):
        """
        Test that the method make_rate_image_from_data_cube works with a custom fit_order.
        """
        custom_fit_order = 2  # Define a custom fitting order
        mock_return_image = np.random.rand(3, 3)  # Create a dummy dark rate image

        # Mock fit_cube to set rate_image in the data_cube
        dark_object_with_data_cube.data_cube.fit_cube = MagicMock()
        dark_object_with_data_cube.data_cube.rate_image = mock_return_image  # Simulate setting the image after fitting

        dark_object_with_data_cube.make_rate_image_from_data_cube(fit_order=custom_fit_order)

        # Assert that fit_cube was called with the custom degree
        dark_object_with_data_cube.data_cube.fit_cube.assert_called_once_with(degree=custom_fit_order)
        assert dark_object_with_data_cube.dark_rate_image is not None  # Ensure dark_rate_image is set
        assert np.array_equal(dark_object_with_data_cube.dark_rate_image,
                              mock_return_image)  # Check the populated image

    @skip_on_github
    def test_make_ma_table_resampled_data_with_read_pattern(self, dark_object_with_data_cube):
        """
        Test the make_ma_table_resampled_data method with a valid read pattern.
        """
        read_pattern = [[1, 2], [3, 4], [5]]
        dark_object_with_data_cube.data_cube.data = np.random.random((5, 4096, 4096))
        dark_object_with_data_cube.data_cube.time_array = np.arange(5)
        dark_object_with_data_cube.make_ma_table_resampled_data(read_pattern=read_pattern)
        assert dark_object_with_data_cube.num_resultants == len(read_pattern)
        assert dark_object_with_data_cube.resampled_data.shape == (3, 4096, 4096)  # 3 resultants
        assert dark_object_with_data_cube.resultant_tau_array.shape == (3,)

        # Check that the data was averaged correctly
        for resultant_i, read_frames in enumerate(read_pattern):
            expected_data = np.mean(dark_object_with_data_cube.data_cube.data[np.array(read_frames) - 1], axis=0)
            np.testing.assert_array_almost_equal(dark_object_with_data_cube.resampled_data[resultant_i], expected_data)

    @skip_on_github
    def test_make_ma_table_resampled_data_even_spacing(self, dark_object_with_data_cube):
        """
        Test the make_ma_table_resampled_data method with num_resultants and num_reads_per_resultant.
        """
        num_resultants = 2
        num_reads_per_resultant = 2
        dark_object_with_data_cube.data_cube.data = np.random.random((4, 4096, 4096))
        dark_object_with_data_cube.data_cube.time_array = np.arange(10)
        dark_object_with_data_cube.make_ma_table_resampled_data(num_resultants=num_resultants,
                                                                num_reads_per_resultant=num_reads_per_resultant)
        assert dark_object_with_data_cube.num_resultants == num_resultants
        assert dark_object_with_data_cube.resampled_data.shape == (num_resultants, 4096, 4096)
        assert dark_object_with_data_cube.resultant_tau_array.shape == (num_resultants,)

        # Check that the data was averaged correctly
        expected_data_1 = np.mean(dark_object_with_data_cube.data_cube.data[0:2], axis=0)
        expected_data_2 = np.mean(dark_object_with_data_cube.data_cube.data[2:4], axis=0)
        np.testing.assert_array_almost_equal(dark_object_with_data_cube.resampled_data[0], expected_data_1)
        np.testing.assert_array_almost_equal(dark_object_with_data_cube.resampled_data[1], expected_data_2)

    def test_make_ma_table_resampled_data_invalid_input(self, dark_object_with_data_cube):
        """
        Test the make_ma_table_resampled_data method with invalid inputs to check for errors.
        """
        with pytest.raises(TypeError):
            dark_object_with_data_cube.make_ma_table_resampled_data(num_resultants="invalid", num_reads_per_resultant=2)
        with pytest.raises(TypeError):
            dark_object_with_data_cube.make_ma_table_resampled_data(num_resultants=None, num_reads_per_resultant=2)
        with pytest.raises(ValueError):
            dark_object_with_data_cube.make_ma_table_resampled_data(num_resultants=1, num_reads_per_resultant=6)

    def test_update_data_quality_array(self, valid_meta_data, valid_ref_type_data_cube, dark_rate_image_3_by_3):
        """
        Test the update_data_quality_array method to ensure that it properly updates
        the DQ array based on the dark_rate_image and threshold values for hot, warm, and dead pixels.
        """

        # Use dqflags.pixel for defining the expected DQ flags
        dqflag_defs = dqflags.pixel
        dark_obj = Dark(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data_cube)
        dark_obj.data_cube.rate_image = dark_rate_image_3_by_3

        # Initialize the smaller mask array to be same as test_dark_rate_image
        dark_obj.dq_mask = np.zeros(dark_rate_image_3_by_3.shape, dtype=np.uint32)

        # Put the dq flags in the dark object.
        dark_obj.dqflag_defs = dqflag_defs

        # Call the update_data_quality_array method with specified thresholds
        dark_obj.update_data_quality_array(hot_pixel_rate=2.0, warm_pixel_rate=1.0, dead_pixel_rate=0.1)

        # Create the expected mask based on the pixel values and threshold comparisons
        expected_mask = np.array([
            [dqflag_defs["HOT"], dqflag_defs["WARM"], dqflag_defs["GOOD"]],
            [dqflag_defs["HOT"], dqflag_defs["WARM"], dqflag_defs["DEAD"]],
            [dqflag_defs["GOOD"], dqflag_defs["DEAD"], dqflag_defs["DEAD"]]
        ], dtype=np.uint32)

        # Assert that the mask array was updated correctly
        np.testing.assert_array_equal(dark_obj.dq_mask, expected_mask,
                                      err_msg="DQ array was not updated as expected.")

    @skip_on_github
    def test_populate_datamodel_tree(self, dark_object_with_data_cube,
                                     valid_ref_type_data_cube,
                                     dark_rate_image_3_by_3):
        """
        Test that the data model tree is correctly populated in the Dark object.
        """
        dark_object_with_data_cube.resampled_data = valid_ref_type_data_cube
        dark_object_with_data_cube.dark_rate_image = dark_rate_image_3_by_3
        dark_object_with_data_cube.dark_rate_image_error = dark_rate_image_3_by_3
        data_model_tree = dark_object_with_data_cube.populate_datamodel_tree()

        # Assuming the Flat data model includes:
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree
        assert 'dark_slope' in data_model_tree
        assert 'dark_slope_error' in data_model_tree
        assert 'dq' in data_model_tree

        # Check the shape and dtype of the 'data' array
        assert data_model_tree['data'].shape == (5, 4096, 4096)
        assert data_model_tree['data'].dtype == np.float32
        # Check the shape and dtype of the 'dark_slope' array
        assert data_model_tree['dark_slope'].shape == (3, 3)
        assert data_model_tree['dark_slope'].dtype == np.float32
        # Check the shape and dtype of the 'dark_slope_error' array
        assert data_model_tree['dark_slope_error'].shape == (3, 3)
        assert data_model_tree['dark_slope_error'].dtype == np.float32
        # Check the shape and dtype of the 'dq' array
        assert data_model_tree['dq'].shape == (4096, 4096)
        assert data_model_tree['dq'].dtype == np.uint32

    def test_dark_outfile_default(self, dark_object_with_data_cube):
        """
        Test that the default outfile name is correct in the Dark object with the assumption
        that the default name is 'roman_dark.asdf'
        """
        assert dark_object_with_data_cube.outfile == "roman_dark.asdf"
