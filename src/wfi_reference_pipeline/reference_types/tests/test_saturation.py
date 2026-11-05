import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_PIX_BORDER_COUNT,
    REF_TYPE_READNOISE,
    REF_TYPE_SATURATION,
    SATURATION_MAX_THRESHOLD,
)
from wfi_reference_pipeline.reference_types.saturation.saturation import Saturation
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for the SAturation class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_SATURATION)
    return test_meta.meta_saturation


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating a valid ref_type_data array (saturation image)."""
    return np.ones((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)).astype(np.float32)


@pytest.fixture
def saturation_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a Mask object with a valid data array."""
    saturation_object_with_data_array = Saturation(meta_data=valid_meta_data,
                                                   ref_type_data=valid_ref_type_data_array)
    yield saturation_object_with_data_array


class TestSaturation:

    def test_saturation_instantiation_with_valid_data(self, saturation_object_with_data_array):
        """
        Test that Saturation object is created successfully with valid input data.
        """
        assert isinstance(saturation_object_with_data_array, Saturation)
        assert saturation_object_with_data_array.saturation_image.shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)

    def test_saturation_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that Saturation raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Saturation(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data_array)

    def test_saturation_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Saturation raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Saturation(meta_data=valid_meta_data, ref_type_data='not_data.txt')

    def test_make_saturation_image(self, saturation_object_with_data_array):
        """
        Test that make_saturation_image sets science pixels to the threshold
        and the 4-pixel reference pixel border to 65535.
        """
        threshold = 60000
        border_value = SATURATION_MAX_THRESHOLD

        saturation_object_with_data_array.make_saturation_image(saturation_threshold=threshold)
        saturation_image = saturation_object_with_data_array.saturation_image

        # Science pixels are set to the requested threshold
        assert np.all(saturation_image[REF_PIX_BORDER_COUNT:-REF_PIX_BORDER_COUNT, REF_PIX_BORDER_COUNT:-REF_PIX_BORDER_COUNT] == threshold)

        # All reference pixels on the border are set to the uint16 maximum
        border_mask = np.ones(saturation_image.shape, dtype=bool)
        border_mask[REF_PIX_BORDER_COUNT:-REF_PIX_BORDER_COUNT, REF_PIX_BORDER_COUNT:-REF_PIX_BORDER_COUNT] = False
        assert np.all(saturation_image[border_mask] == border_value)

    def test_make_saturation_image_with_invalid_input(self, saturation_object_with_data_array):
        """
        Test that the make_saturation_image method handles invalid inputs.
        """
        # Check for invalid string input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object_with_data_array.make_saturation_image(saturation_threshold='invalid_value')

        # Check for negative threshold
        with pytest.raises(ValueError, match="Saturation threshold must be 1 or less than unit16 maximum value 66535."):
            saturation_object_with_data_array.make_saturation_image(saturation_threshold=-1)

        # Check for greater than uint16 value
        with pytest.raises(ValueError, match="Saturation threshold must be 1 or less than unit16 maximum value 66535."):
            saturation_object_with_data_array.make_saturation_image(saturation_threshold=700000)

        # Check for invalid list input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object_with_data_array.make_saturation_image(saturation_threshold=[55000])

        # Check for invalid dictionary input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object_with_data_array.make_saturation_image(saturation_threshold={'value': 55000})

    def test_populate_datamodel_tree(self, saturation_object_with_data_array):
        """
        Test that the populate_datamodel_tree method constructs the data model correctly.
        """
        saturation_object_with_data_array.make_saturation_image()
        data_model_tree = saturation_object_with_data_array.populate_datamodel_tree()

        # Assuming the Saturation data model includes:
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree
        assert 'dq' in data_model_tree

        # Check the shape and dtype of the 'data' array
        assert data_model_tree['data'].shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert data_model_tree['data'].dtype == np.float32

        # Check the shape and dtype of the 'dq' array
        assert data_model_tree['dq'].shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert data_model_tree['dq'].dtype == np.uint32

    def test_saturation_outfile_default(self, saturation_object_with_data_array):
        """
        Test that the default outfile name is correct in the Saturation object with the assumption
        that the default name is 'roman_saturation.asdf'
        """
        assert saturation_object_with_data_array.outfile == "roman_saturation.asdf"
