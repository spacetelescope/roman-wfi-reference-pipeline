import pytest
import numpy as np
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_MASK, REF_TYPE_READNOISE


@pytest.fixture
def valid_meta_data():
    """
    Fixture to create a valid mask data array (4096x4096, uint32).
    """
    test_meta = MakeTestMeta(ref_type=REF_TYPE_MASK)
    return test_meta.meta_mask


@pytest.fixture
def valid_ref_type_data():
    """
    Fixture to create a valid mask data array (4096x4096, uint32).
    """
    return np.zeros((4096, 4096), dtype=np.uint32)


@pytest.fixture
def mask_object(valid_meta_data, valid_ref_type_data):
    """
    Fixture to create a Mask object with valid reference type data.
    """
    obj = Mask(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    yield obj


class TestMask:

    def test_mask_instantiation_with_valid_data(self, mask_object):
        """
        Test that Mask object is created successfully with valid input data.
        """
        assert isinstance(mask_object, Mask)
        assert mask_object.mask_image.shape == (4096, 4096)
        assert mask_object.mask_image.dtype == np.uint32
        assert mask_object.mask_image.dtype == np.uint32

    def test_mask_instantiation_with_invalid_metadata(self, valid_ref_type_data):
        """
        Test that Mask raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Mask(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data)

    def test_mask_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Mask raises ValueError with invalid reference type data.
        """
        with pytest.raises(ValueError):
            Mask(meta_data=valid_meta_data, ref_type_data=np.ones((10, 10), dtype=np.float32))

    def test_make_mask_image(self, mask_object):
        """
        Test that the make_mask_image method successfully creates the mask image.
        """
        mask_object.make_mask_image()
        assert mask_object.mask_image is not None

    def test_update_mask_ref_pixels(self, mask_object):
        """
        Test that the reference pixels are correctly flagged by _update_mask_ref_pixels.
        """
        mask_object.make_mask_image()
        top_pixels = mask_object.mask_image[:4, :]
        bottom_pixels = mask_object.mask_image[-4:, :]
        left_pixels = mask_object.mask_image[:, :4]
        right_pixels = mask_object.mask_image[:, -4:]

        assert np.all(top_pixels == 2**31)
        assert np.all(bottom_pixels == 2**31)
        assert np.all(left_pixels == 2**31)
        assert np.all(right_pixels == 2**31)

    def test_populate_datamodel_tree(self, mask_object):
        """
        Test that the data model tree is correctly populated.
        """
        data_model_tree = mask_object.populate_datamodel_tree()
        assert 'meta' in data_model_tree
        assert 'dq' in data_model_tree
        assert data_model_tree['dq'].shape == (4096, 4096)
        assert data_model_tree['dq'].dtype == np.uint32

    def test_mask_outfile_default(self, mask_object):
        """
        Test that the default outfile name is correct.
        """
        assert mask_object.outfile == "roman_mask.asdf"
