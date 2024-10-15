import pytest
import numpy as np
from unittest.mock import MagicMock
from wfi_reference_pipeline.reference_types.saturation.saturation import Saturation
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_SATURATION, REF_TYPE_READNOISE
from astropy import units as u


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Saturation class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_SATURATION)
    return test_meta.meta_saturation


@pytest.fixture
def valid_ref_type_data():
    """Fixture for generating valid ref_type_data (saturation data)."""
    # Simulating a 2D saturation data array (4096x4096)
    ref_type_data = np.random.randint(0, 100000, size=(4096, 4096)).astype(np.float32)
    return ref_type_data


@pytest.fixture
def saturation_object(valid_meta_data, valid_ref_type_data):
    """Fixture for initializing a valid Saturation object."""
    saturation_obj = Saturation(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    saturation_obj.make_saturation_image()
    yield saturation_obj


class TestSaturation:

    def test_saturation_instantiation_with_valid_data(self, saturation_object):
        """
        Test that Saturation object is created successfully with valid input data.
        """
        assert isinstance(saturation_object, Saturation)
        assert saturation_object.saturation_image.shape == (4096, 4096)

    def test_saturation_instantiation_with_invalid_metadata(self, valid_ref_type_data):
        """
        Test that Saturation raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)  # Assume this creates an invalid type
        with pytest.raises(TypeError):
            Saturation(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data)

    def test_saturation_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Saturation raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Saturation(meta_data=valid_meta_data, ref_type_data='not_data.txt')

    def test_make_saturation_image(self, saturation_object):
        """
        Test that the make_saturation_image method creates an image of the correct uniform value.
        """
        saturation_object.make_saturation_image(saturation_threshold=60000)
        assert np.all(saturation_object.saturation_image == 60000)

    def test_make_saturation_image_with_invalid_input(self, saturation_object):
        """
        Test that the make_saturation_image method handles invalid inputs.
        """
        # Check for invalid string input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object.make_saturation_image(saturation_threshold='invalid_value')

        # Check for negative threshold
        with pytest.raises(ValueError,
                           match="Saturation threshold must be positive and less than uint16 maximum allowed value."):
            saturation_object.make_saturation_image(saturation_threshold=-1)

        # Check for invalid list input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object.make_saturation_image(saturation_threshold=[55000])

        # Check for invalid dictionary input
        with pytest.raises(ValueError, match="Saturation threshold must be a float or an int."):
            saturation_object.make_saturation_image(saturation_threshold={'value': 55000})

    # def test_update_data_quality_array(self, saturation_object):
    #     """
    #     Test that the update_data_quality_array method flags the correct pixels.
    #     """
    #     # Initialize mask with zeros
    #     saturation_object.mask = np.zeros_like(saturation_object.saturation_image, dtype=int)
    #     saturation_object.saturation_image = np.random.randint(0, 100000, size=(4096, 4096)).astype(np.float32)
    #
    #     saturation_object.update_data_quality_array(bad_saturation_threshold=64000)
    #     assert np.any(saturation_object.mask[saturation_object.saturation_image > 64000] == saturation_object.dqflag_defs['NO_SAT_CHECK'])

    def test_populate_datamodel_tree(self, saturation_object):
        """
        Test that the populate_datamodel_tree method constructs the data model correctly.
        """
        saturation_datamodel_tree = saturation_object.populate_datamodel_tree()
        assert 'meta' in saturation_datamodel_tree
        assert 'data' in saturation_datamodel_tree
        assert 'dq' in saturation_datamodel_tree
        assert saturation_datamodel_tree['data'].shape == (4096, 4096)

