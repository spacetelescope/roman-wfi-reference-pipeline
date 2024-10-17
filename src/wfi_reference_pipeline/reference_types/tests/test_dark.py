import pytest
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_DARK, REF_TYPE_READNOISE
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from unittest.mock import MagicMock
import numpy as np
from romancal.lib import dqflags


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Dark class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
    return test_meta.meta_dark


@pytest.fixture
def valid_ref_type_data():
    """Fixture for generating valid ref_type_data (read cube)."""
    ref_type_data, _ = simulate_dark_reads(5)  # Simulate a 5-read data cube
    return ref_type_data


@pytest.fixture
def dark_object(valid_meta_data, valid_ref_type_data):
    """Fixture for initializing a valid Dark object."""
    dark_obj = Dark(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    dark_obj.make_rate_image_from_data_cube()
    dark_obj.make_ma_table_resampled_data(num_resultants=3, num_reads_per_resultant=1)
    yield dark_obj


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

    def test_dark_instantiation_with_valid_data(self, dark_object):
        """
        Test that Dark object is created successfully with valid input data.
        """
        assert isinstance(dark_object, Dark)
        assert dark_object.data_cube.data.shape == (5, 4096, 4096)

    def test_dark_instantiation_with_invalid_metadata(self, valid_ref_type_data):
        """
        Test that Mask raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Dark(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data)

    def test_dark_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Mask raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Dark(meta_data=valid_meta_data, ref_type_data='not_data.txt')

    def test_make_rate_image_default_fit_order(self, dark_object):
        """
        Test that the method make_rate_image_from_data_cube works with default fit_order.
        """
        # Mock the fit_cube method to ensure it is called with correct arguments
        dark_object.data_cube.fit_cube = MagicMock()
        dark_object.make_rate_image_from_data_cube()
        # Assert that fit_cube was called with degree=1 (default fit_order)
        dark_object.data_cube.fit_cube.assert_called_once_with(degree=1)
        assert dark_object.dark_rate_image is not None
        assert dark_object.dark_rate_image_error is not None

    def test_make_rate_image_custom_fit_order(self, dark_object):
        """
        Test that make_rate_image_from_data_cube works with a custom fit_order.
        """
        # Mock the fit_cube method to ensure it is called with correct arguments
        dark_object.data_cube.fit_cube = MagicMock()
        # Call the method with a custom fit_order (e.g., 2)
        dark_object.make_rate_image_from_data_cube(fit_order=2)
        # Assert that fit_cube was called with degree=2
        dark_object.data_cube.fit_cube.assert_called_once_with(degree=2)
        assert dark_object.dark_rate_image is not None
        assert dark_object.dark_rate_image_error is not None

    def test_make_ma_table_resampled_data_with_read_pattern(self, dark_object):
        """
        Test the make_ma_table_resampled_data method with a valid read pattern.
        """
        read_pattern = [[1, 2], [3, 4], [5]]
        dark_object.data_cube.data = np.random.random((5, 4096, 4096))
        dark_object.data_cube.time_array = np.arange(5)
        dark_object.make_ma_table_resampled_data(read_pattern=read_pattern)
        assert dark_object.num_resultants == len(read_pattern)
        assert dark_object.resampled_data.shape == (3, 4096, 4096)  # 3 resultants
        assert dark_object.resultant_tau_array.shape == (3,)

        # Check that the data was averaged correctly
        for resultant_i, read_frames in enumerate(read_pattern):
            expected_data = np.mean(dark_object.data_cube.data[np.array(read_frames) - 1], axis=0)
            np.testing.assert_array_almost_equal(dark_object.resampled_data[resultant_i], expected_data)

    def test_make_ma_table_resampled_data_even_spacing(self, dark_object):
        """
        Test the make_ma_table_resampled_data method with num_resultants and num_reads_per_resultant.
        """
        num_resultants = 2
        num_reads_per_resultant = 2
        dark_object.data_cube.data = np.random.random((4, 4096, 4096))
        dark_object.data_cube.time_array = np.arange(10)
        dark_object.make_ma_table_resampled_data(num_resultants=num_resultants,
                                                 num_reads_per_resultant=num_reads_per_resultant)
        assert dark_object.num_resultants == num_resultants
        assert dark_object.resampled_data.shape == (num_resultants, 4096, 4096)
        assert dark_object.resultant_tau_array.shape == (num_resultants,)

        # Check that the data was averaged correctly
        expected_data_1 = np.mean(dark_object.data_cube.data[0:2], axis=0)
        expected_data_2 = np.mean(dark_object.data_cube.data[2:4], axis=0)
        np.testing.assert_array_almost_equal(dark_object.resampled_data[0], expected_data_1)
        np.testing.assert_array_almost_equal(dark_object.resampled_data[1], expected_data_2)

    def test_make_ma_table_resampled_data_invalid_input(self, dark_object):
        """
        Test the make_ma_table_resampled_data method with invalid inputs to check for errors.
        """
        with pytest.raises(TypeError):
            dark_object.make_ma_table_resampled_data(num_resultants="invalid", num_reads_per_resultant=2)
        with pytest.raises(TypeError):
            dark_object.make_ma_table_resampled_data(num_resultants=None, num_reads_per_resultant=2)
        with pytest.raises(ValueError):
            dark_object.make_ma_table_resampled_data(num_resultants=1, num_reads_per_resultant=6)

    def test_update_data_quality_array(self, valid_meta_data, valid_ref_type_data, dark_rate_image_3_by_3):
        """
        Test the update_data_quality_array method to ensure that it properly updates
        the DQ array based on the dark_rate_image and threshold values for hot, warm, and dead pixels.
        """

        # Use dqflags.pixel for defining the expected DQ flags
        dqflag_defs = dqflags.pixel
        dark_obj = Dark(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
        dark_obj.data_cube.rate_image = dark_rate_image_3_by_3

        # Initialize the smaller mask array to be same as test_dark_rate_image
        dark_obj.mask = np.zeros(dark_rate_image_3_by_3.shape, dtype=np.uint32)

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
        np.testing.assert_array_equal(dark_obj.mask, expected_mask,
                                      err_msg="DQ array was not updated as expected.")

