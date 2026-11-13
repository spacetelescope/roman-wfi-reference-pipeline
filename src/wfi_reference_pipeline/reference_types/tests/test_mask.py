import numpy as np
import pytest
from roman_datamodels.dqflags import pixel as dqflags

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_MASK,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_meta import MakeMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for the Mask class."""
    test_meta = MakeMeta(ref_type=REF_TYPE_MASK, instrument_detector="WFI01")
    return test_meta.meta_mask

@pytest.fixture
def valid_input_user_mask_array():
    """Fixture for generating a valid input_user_mask array (mask image)."""
    arr = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)  # Simulate a valid mask image
    arr[1000:1100, 1000:1100] |= dqflags.OTHER_BAD_PIXEL.value
    return arr

@pytest.fixture
def mask_object_with_data_array(valid_meta_data, valid_input_user_mask_array):
    """Fixture for initializing a Mask object with a valid data array."""
    mask_object_with_data_array = Mask(meta_data=valid_meta_data,
                                       input_user_mask=valid_input_user_mask_array)
    yield mask_object_with_data_array

@pytest.fixture
def fake_dark_filelist():
    """Fixture for generating a fake dark filelist."""
    return [f"fake_prepped_dark_{i}.asdf" for i in range(5)]

@pytest.fixture
def fake_flat_filelist():
    """Fixture for generating a fake flat filelist."""
    return [f"fake_prepped_flat_{i}.asdf" for i in range(5)]

@pytest.fixture
def fake_invalid_filelist():
    """Fixture for generating a filelist of invalid files."""
    return [f"bad_random_file_{i}.asdf" for i in range(5)]

@pytest.fixture
def fake_superdark_array():
    """Fixture for a fake superdark cube returned by the stubbed prep_superdark()."""
    nreads = 10
    return np.zeros((nreads, DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.float32)

@pytest.fixture
def fake_super_rate_array():
    """Fixture for a fake super rate image returned by the stubbed prep_super_rate()."""
    return np.ones((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.float32)


class TestMask:

    def test_mask_instantiation_with_valid_input_user_mask_array(self, mask_object_with_data_array):
        """
        Test that Mask object is created successfully with valid input data array.
        """
        assert isinstance(mask_object_with_data_array, Mask)
        assert mask_object_with_data_array.mask_image.shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert mask_object_with_data_array.mask_image.dtype == np.uint32
        assert np.count_nonzero((mask_object_with_data_array.mask_image & dqflags.OTHER_BAD_PIXEL.value) != 0)

    def test_mask_instantiation_with_invalid_metadata(self, valid_input_user_mask_array):
        """
        Test that Mask raises ValueError with invalid metadata type.
        """
        bad_test_meta = MakeMeta(ref_type=REF_TYPE_READNOISE, instrument_detector="WFI01")
        with pytest.raises(ValueError):
            Mask(meta_data=bad_test_meta.meta_readnoise, input_user_mask=valid_input_user_mask_array)

    def test_mask_instantiation_with_invalid_input_user_mask(self, valid_meta_data):
        """
        Test that Mask raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Mask(meta_data=valid_meta_data, input_user_mask="invalid_input_mask")

    def test_mask_instantiation_with_wrong_input_user_mask(self, valid_meta_data):
        """
        Test that Mask raises ValueError with array of wrong dimensions.
        """
        with pytest.raises(TypeError):
            Mask(meta_data=valid_meta_data, input_user_mask=np.ones((10, 10)).astype(np.float32))

    def test_make_mask_image_with_data_array(self, mask_object_with_data_array):
        """
        Test that the make_mask_image method successfully creates the mask image.
        """
        mask_object_with_data_array.make_mask_image()
        assert mask_object_with_data_array.mask_image is not None

    def test_update_mask_ref_pixels(self, mask_object_with_data_array):
        """
        Test that the reference pixels are correctly flagged by _update_mask_ref_pixels.
        """
        mask_object_with_data_array.make_mask_image()
        top_pixels = mask_object_with_data_array.mask_image[:4, :]
        bottom_pixels = mask_object_with_data_array.mask_image[-4:, :]
        left_pixels = mask_object_with_data_array.mask_image[:, :4]
        right_pixels = mask_object_with_data_array.mask_image[:, -4:]

        assert np.all(top_pixels == dqflags.REFERENCE_PIXEL.value)
        assert np.all(bottom_pixels == dqflags.REFERENCE_PIXEL.value)
        assert np.all(left_pixels == dqflags.REFERENCE_PIXEL.value)
        assert np.all(right_pixels == dqflags.REFERENCE_PIXEL.value)

    def test_populate_datamodel_tree(self, mask_object_with_data_array):
        """
        Test that the data model tree is correctly populated in the Mask object.
        """
        data_model_tree = mask_object_with_data_array.populate_datamodel_tree()

        # Assuming the Mask data model includes:
        assert 'meta' in data_model_tree
        assert 'dq' in data_model_tree

        # Check the shape and dtype of the 'dq' array
        assert data_model_tree['dq'].shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert data_model_tree['dq'].dtype == np.uint32

    def test_mask_outfile_default(self, mask_object_with_data_array):
        """
        Test that the default outfile name is correct.
        """
        assert mask_object_with_data_array.outfile == "roman_mask.asdf"

    def test_one_dark_filelist(self, valid_meta_data, fake_dark_filelist, fake_superdark_array):
        """
        Test that a Mask object created with only a dark_filelist populates
        self.superdark and leaves super_rate_image / mask_image unset.
        """
        mask_obj = Mask(
            meta_data=valid_meta_data,
            dark_filelist=fake_dark_filelist,
            input_super_dark=fake_superdark_array,
        )
        assert mask_obj.super_dark is fake_superdark_array
        assert mask_obj.super_rate is None

    def test_one_flat_filelist(self, valid_meta_data, fake_flat_filelist, fake_super_rate_array):
        """
        Test that a Mask object created with only a flat_filelist populates
        self.super_rate_image and leaves superdark / mask_image unset.
        """
        mask_obj = Mask(
            meta_data=valid_meta_data,
            flat_filelist=fake_flat_filelist,
            input_super_rate=fake_super_rate_array,
        )
        assert mask_obj.super_rate is fake_super_rate_array
        assert mask_obj.super_dark is None

    def test_input_user_mask_array_only(self, mask_object_with_data_array):
        """
        Test that a Mask object created with only input_user_mask populates
        mask_image and leaves superdark / super_rate_image unset.
        """
        assert mask_object_with_data_array.mask_image is not None
        assert mask_object_with_data_array.super_dark is None
        assert mask_object_with_data_array.super_rate is None

    def test_two_filelists(self, valid_meta_data, fake_dark_filelist, fake_flat_filelist,
                            fake_superdark_array, fake_super_rate_array):
        """
        Test that a Mask object created with both dark_filelist and
        flat_filelist populates both superdark and super_rate_image.
        """
        mask_obj = Mask(
            meta_data=valid_meta_data,
            dark_filelist=fake_dark_filelist,
            flat_filelist=fake_flat_filelist,
            input_super_dark=fake_superdark_array,
            input_super_rate=fake_super_rate_array,
        )
        assert mask_obj.super_dark is fake_superdark_array
        assert mask_obj.super_rate is fake_super_rate_array

    def test_two_filelists_and_input_user_mask_propagates(
        self, valid_meta_data, fake_dark_filelist, fake_flat_filelist,
        fake_superdark_array, fake_super_rate_array,
    ):
        """
        Test that input_user_mask is correctly propagated into mask_image
        additively when both filelists are also supplied.
        A block well away from the 4px border is seeded with REFERENCE_PIXEL
        so it can only have come from input_user_mask, not update_mask_ref_pixels().
        """
        seeded_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)
        seeded_mask[100:110, 100:110] = dqflags.REFERENCE_PIXEL.value

        mask_obj = Mask(
            meta_data=valid_meta_data,
            dark_filelist=fake_dark_filelist,
            flat_filelist=fake_flat_filelist,
            input_user_mask=seeded_mask,
            input_super_dark=fake_superdark_array,
            input_super_rate=fake_super_rate_array,
        )
        mask_obj.make_mask_image()

        assert np.all(mask_obj.mask_image[100:110, 100:110] == dqflags.REFERENCE_PIXEL.value)
        assert np.all(mask_obj.mask_image[:4, :] == dqflags.REFERENCE_PIXEL.value)
        assert np.all(mask_obj.mask_image[-4:, :] == dqflags.REFERENCE_PIXEL.value)

    def test_no_input_raises(self, valid_meta_data):
        """
        Test that Mask raises ValueError when no input_user_mask, super dark,
        super rate image, or dark/flat filelist is supplied.
        """
        with pytest.raises(ValueError):
            Mask(meta_data=valid_meta_data)