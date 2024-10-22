import pytest
import numpy as np
from astropy import units as u
from wfi_reference_pipeline.reference_types.gain.gain import Gain
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_GAIN, REF_TYPE_READNOISE


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaGain metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_GAIN)
    return test_meta.meta_gain


@pytest.fixture
def valid_ref_type_data():
    """Fixture for generating valid reference type data (gain image)."""
    return np.random.random((4096, 4096))  # Simulate a valid 2D gain image


@pytest.fixture
def gain_object(valid_meta_data, valid_ref_type_data):
    """Fixture for initializing a valid Gain object."""
    obj = Gain(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    yield obj


class TestGain:

    def test_gain_instantiation_with_valid_data(self, gain_object):
        """
        Test that Gain object is created successfully with valid input data.
        """
        assert isinstance(gain_object, Gain)
        assert gain_object.gain_image.shape == (4096, 4096)

    def test_gain_instantiation_with_invalid_metadata(self, valid_ref_type_data):
        """
        Test that Gain raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Gain(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data)

    def test_gain_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Gain raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Gain(meta_data=valid_meta_data, ref_type_data="invalid_data")

    def test_gain_instantiation_with_quantity_object(self, valid_meta_data):
        """
        Test that Gain object handles Quantity object as ref_type_data correctly.
        """
        data_quantity = (np.random.random((4096, 4096)) * u.electron / u.DN)  # Quantity object
        gain_obj = Gain(meta_data=valid_meta_data, ref_type_data=data_quantity)

        assert gain_obj.gain_image is not None
        assert isinstance(gain_obj.gain_image, np.ndarray)
        assert gain_obj.gain_image.shape == (4096, 4096)

    def test_populate_datamodel_tree(self, gain_object):
        """
        Test that the data model tree is correctly populated in the Gain object.
        """
        data_model_tree = gain_object.populate_datamodel_tree()

        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree
        assert data_model_tree['data'].shape == (4096, 4096)
        assert data_model_tree['data'].unit == u.electron / u.DN

    def test_gain_outfile_default(self, gain_object):
        """
        Test that the default outfile name is correct.
        """
        assert gain_object.outfile == "roman_gain.asdf"
