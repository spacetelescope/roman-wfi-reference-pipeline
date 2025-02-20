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
def valid_ref_type_data_array():
    """Fixture for generating valid reference type data (gain image)."""
    return np.random.random((4096, 4096))  # Simulate a valid 2D gain image


@pytest.fixture
def gain_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a Gain object with valid data."""
    gain_object_with_data_array = Gain(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data_array)
    yield gain_object_with_data_array


class TestGain:

    def test_gain_instantiation_with_valid_ref_type_data(self, gain_object_with_data_array):
        """
        Test that Gain object is created successfully with valid input data.
        """
        assert isinstance(gain_object_with_data_array, Gain)
        assert gain_object_with_data_array.gain_image.shape == (4096, 4096)

    def test_gain_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that Gain raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Gain(meta_data=bad_test_meta.meta_readnoise, ref_type_data=valid_ref_type_data_array)

    def test_gain_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that Gain raises ValueError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Gain(meta_data=valid_meta_data, ref_type_data="invalid_ref_data")

    def test_populate_datamodel_tree(self, gain_object_with_data_array):
        """
        Test that the data model tree is correctly populated in the Gain object.
        """
        data_model_tree = gain_object_with_data_array.populate_datamodel_tree()

        # Assuming that gain data model has meta and data.
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree

        # Check data shape and type.
        assert data_model_tree['data'].shape == (4096, 4096)
        assert data_model_tree['data'].dtype == np.float32

    def test_gain_outfile_default(self, gain_object_with_data_array):
        """
        Test that the default outfile name is correct.
        """
        assert gain_object_with_data_array.outfile == "roman_gain.asdf"