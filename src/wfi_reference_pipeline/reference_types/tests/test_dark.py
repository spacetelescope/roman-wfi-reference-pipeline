import os

import numpy as np
import pytest
from roman_datamodels import dqflags

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_DARK,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_meta import MakeMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Dark class."""
    test_meta = MakeMeta(ref_type=REF_TYPE_DARK, instrument_detector="WFI01")
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
                    hot, warm, good,
                    good, good, good]
    """
    return np.array([
        [2.1, 1.1, 0.02],  # should return hot, warm, no flag set
        [2.0, 1.0, 0.01],  # should return hot, warm, no flag set
        [0.05, 0.01, 0.01],  # should return no flag set, no flag set, no flag set
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
        bad_test_meta = MakeMeta(ref_type=REF_TYPE_READNOISE, instrument_detector="WFI01")
        with pytest.raises(TypeError):
            Dark(meta_data=bad_test_meta.meta_readnoise, ref_type_data=dark_object_with_data_cube)

    def test_dark_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Flat raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Dark(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')

    def test_make_rate_image_from_data_cube_default_fit_order(self, dark_object_with_data_cube, mocker):
        """
        Test that the method make_rate_image_from_data_cube works with default fit_order.
        """
        mock_return_image = np.random.rand(3, 3)  # Create a dummy dark rate image
        mock_return_error_image = np.random.rand(3, 3)  # Dummy error image
        dark_object_with_data_cube.data_cube.fit_cube = mocker.MagicMock()
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

    def test_make_rate_image_from_data_cube_custom_fit_order(self, dark_object_with_data_cube, mocker):
        """
        Test that the method make_rate_image_from_data_cube works with a custom fit_order.
        """
        custom_fit_order = 2  # Define a custom fitting order
        mock_return_image = np.random.rand(3, 3)  # Create a dummy dark rate image

        # Mock fit_cube to set rate_image in the data_cube
        dark_object_with_data_cube.data_cube.fit_cube = mocker.MagicMock()
        dark_object_with_data_cube.data_cube.rate_image = mock_return_image  # Simulate setting the image after fitting

        dark_object_with_data_cube.make_rate_image_from_data_cube(fit_order=custom_fit_order)

        # Assert that fit_cube was called with the custom degree
        dark_object_with_data_cube.data_cube.fit_cube.assert_called_once_with(degree=custom_fit_order)
        assert dark_object_with_data_cube.dark_rate_image is not None  # Ensure dark_rate_image is set
        assert np.array_equal(dark_object_with_data_cube.dark_rate_image,
                              mock_return_image)  # Check the populated image

    def test_update_data_quality_array(self, valid_meta_data, valid_ref_type_data_cube, dark_rate_image_3_by_3):
        """
        Test the update_data_quality_array method to ensure that it properly updates
        the DQ array based on the dark_rate_image and threshold values for hot, warm, and dead pixels.
        """

        # Use dqflags.pixel for defining the expected DQ flags
        dqflag_defs = dqflags.pixel
        dark_obj = Dark(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data_cube)
        dark_obj.dark_rate_image = dark_rate_image_3_by_3

        # Initialize the smaller mask array to be same as test_dark_rate_image
        dark_obj.dq_mask = np.zeros(dark_rate_image_3_by_3.shape, dtype=np.uint32)

        # Put the dq flags in the dark object.
        dark_obj.dqflag_defs = dqflag_defs

        # Call the update_data_quality_array method with specified thresholds
        dark_obj.update_data_quality_array(2.0, 1.0)

        # Create the expected mask based on the pixel values and threshold comparisons
        expected_mask = np.array([
            [dqflag_defs["HOT"], dqflag_defs["WARM"], dqflag_defs["GOOD"]],
            [dqflag_defs["HOT"], dqflag_defs["WARM"], dqflag_defs["GOOD"]],
            [dqflag_defs["GOOD"], dqflag_defs["GOOD"], dqflag_defs["GOOD"]]
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
        dark_object_with_data_cube.dark_rate_image = dark_rate_image_3_by_3
        dark_object_with_data_cube.dark_rate_image_error = dark_rate_image_3_by_3
        data_model_tree = dark_object_with_data_cube.populate_datamodel_tree()

        # Assuming the Flat data model includes:
        assert 'meta' in data_model_tree
        assert 'dark_slope' in data_model_tree
        assert 'dark_slope_error' in data_model_tree
        assert 'dq' in data_model_tree

        # Check the shape and dtype of the 'dark_slope' array
        assert data_model_tree['dark_slope'].shape == (3, 3)
        assert data_model_tree['dark_slope'].dtype == np.float32
        # Check the shape and dtype of the 'dark_slope_error' array
        assert data_model_tree['dark_slope_error'].shape == (3, 3)
        assert data_model_tree['dark_slope_error'].dtype == np.float32
        # Check the shape and dtype of the 'dq' array
        assert data_model_tree['dq'].shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert data_model_tree['dq'].dtype == np.uint32

    def test_dark_outfile_default(self, dark_object_with_data_cube):
        """
        Test that the default outfile name is correct in the Dark object with the assumption
        that the default name is 'roman_dark.asdf'
        """
        assert dark_object_with_data_cube.outfile == "roman_dark.asdf"
