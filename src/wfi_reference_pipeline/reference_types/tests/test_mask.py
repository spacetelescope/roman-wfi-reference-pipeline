import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_MASK,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for the Mask class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_MASK)
    return test_meta.meta_mask


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating a valid ref_type_data array (mask image)."""
    return np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)  # Simulate a valid mask image

@pytest.fixture
def mask_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a Mask object with a valid data array."""
    mask_object_with_data_array = Mask(meta_data=valid_meta_data,
                                       ref_type_data=valid_ref_type_data_array)
    yield mask_object_with_data_array

class TestMask:

    def test_mask_instantiation_with_valid_ref_type_data_array(self, mask_object_with_data_array):
        """
        Test that Mask object is created successfully with valid input data array.
        """
        assert isinstance(mask_object_with_data_array, Mask)
        assert mask_object_with_data_array.mask_image.shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
        assert mask_object_with_data_array.mask_image.dtype == np.uint32

    def test_mask_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that Mask raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Mask(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data_array)

    def test_mask_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Mask raises ValueError with invalid reference type data.
        """
        with pytest.raises(ValueError):
            Mask(meta_data=valid_meta_data, ref_type_data="invalid_ref_data")

    def test_mask_instantiation_with_wrong_ref_type_data(self, valid_meta_data):
        """
        Test that Mask raises ValueError with reference type data.
        """
        with pytest.raises(ValueError):
            Mask(meta_data=valid_meta_data, ref_type_data=np.ones((10, 10)).astype(np.float32))

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

        assert np.all(top_pixels == 2**31)
        assert np.all(bottom_pixels == 2**31)
        assert np.all(left_pixels == 2**31)
        assert np.all(right_pixels == 2**31)

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

    def test_mask_object_from_valid_filelist(self):
        """
        Test that a Mask object is able to correctly initialize when receiving file_list
        of valid prepped flats, prepped darks, and unprepped other files.
        """
        nfiles = 15

        # Only need to check the paths, not create actual files
        fake_flats = [f"fake_prepped_flat_{i}.asdf" for i in range(nfiles)]
        fake_darks = [f"fake_prepped_dark_{i}.asdf" for i in range(nfiles)]

        # Putting them in a single filelist arr to pass to Mask
        fake_filelist = fake_flats + fake_darks

        test_meta = MakeTestMeta(ref_type=REF_TYPE_MASK)
        rfp_mask = Mask(meta_data=test_meta.meta_mask,
                        file_list=fake_filelist)
        
        assert(sorted(fake_flats) == sorted(rfp_mask.flat_filelist))
        assert(sorted(fake_darks) == sorted(rfp_mask.dark_filelist))

    def test_mask_object_from_bad_filelist(self):
        """
        Test that a RuntimeError is raised if no valid flat or dark files were sorted.
        """
        # Only need to check the paths, not create actual files
        nfiles = 15
        fake_flats = [f"fake_prepped_flat_{i}.asdf" for i in range(nfiles)]
        fake_darks = [f"fake_prepped_dark_{i}.asdf" for i in range(nfiles)]
        invalid_files = [f"bad_random_file_{i}.asdf" for i in range(nfiles)]

        filelist_with_invalid_files = fake_flats + fake_darks + invalid_files

        test_meta = MakeTestMeta(ref_type=REF_TYPE_MASK)

        with pytest.raises(ValueError):
            Mask(meta_data=test_meta.meta_mask,
                 file_list=filelist_with_invalid_files)
