import pytest
import numpy as np
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_FLAT, REF_TYPE_READNOISE


@pytest.fixture
def valid_meta_data():
    """Fixture to provide valid meta data."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_FLAT)
    return test_meta.meta_flat


@pytest.fixture
def valid_ref_type_data():
    """Fixture to provide valid 2D data array."""
    return np.ones((4088, 4088)).astype(np.float32)


@pytest.fixture
def valid_data_cube():
    """Fixture to provide valid 3D data cube."""
    return np.ones((5, 4088, 4088)).astype(np.float32)


@pytest.fixture
def flat_object(valid_meta_data, valid_ref_type_data):
    """Fixture to create a valid Flat instance with 2D data."""
    return Flat(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)


@pytest.fixture
def flat_object_from_cube(valid_meta_data, valid_data_cube):
    """Fixture to create a valid Flat instance with 2D data."""
    return Flat(meta_data=valid_meta_data, ref_type_data=valid_data_cube)


class TestFlat:

    def test_flat_instantiation_with_valid_meta_data(self, flat_object):
        """
        Test that Flat object is created successfully with valid input data.
        """
        assert isinstance(flat_object, Flat)
        assert flat_object.flat_image.shape == (4088, 4088)
        assert flat_object.flat_image.dtype == np.float32

    def test_flat_instantiation_with_invalid_metadata_reftype(self, valid_ref_type_data):
        """
        Test that Flat raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Flat(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data)

    def test_flat_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Flat raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Flat(meta_data=valid_meta_data, ref_type_data='not_data.txt')

    def test_make_flat_image_with_data_cube(self, valid_meta_data, valid_data_cube):
        """
        Test that the make_mask_image method successfully creates the mask image.
        """
        flat_object = Flat(meta_data=valid_meta_data, ref_type_data=valid_data_cube)
        flat_object.make_flat_image()
        assert flat_object.flat_image is not None

    def test_calculate_error_with_default_array(self, flat_object_from_cube):
        """
        Test calculate_error with no input array.
        """
        flat_object_from_cube.calculate_error()
        assert flat_object_from_cube.flat_error is not None
        assert flat_object_from_cube.flat_error.shape == (4088, 4088)

    def test_calculate_error_with_custom_array(self, flat_object_from_cube):
        """
        Test calculate_error with a user-supplied error array.
        """
        custom_error = np.ones((4088, 4088), dtype=np.float32) * 0.05
        flat_object_from_cube.calculate_error(error_array=custom_error)
        assert np.array_equal(flat_object_from_cube.flat_error, custom_error)

    def test_update_data_quality_array(self, flat_object):
        """
        Test update_data_quality_array adds DQ flags for low QE pixels.
        """
        low_qe_threshold = 0.2
        flat_object.flat_image[2000, 2000] = 0.1  # Simulate a low-QE pixel
        flat_object.update_data_quality_array(low_qe_threshold=low_qe_threshold)
        assert flat_object.mask[2000, 2000] > 0

    def test_populate_datamodel_tree(self, flat_object_from_cube):
        """Test populate_datamodel_tree method to check data model creation."""
        flat_object_from_cube.make_flat_image()
        data_model = flat_object_from_cube.populate_datamodel_tree()
        assert data_model['data'].shape == (4088, 4088)
        assert data_model['err'].shape == (4088, 4088)
        assert data_model['dq'].shape == (4088, 4088)
