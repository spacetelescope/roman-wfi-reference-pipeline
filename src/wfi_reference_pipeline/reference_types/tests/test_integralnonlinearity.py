import numpy as np
import pytest

from wfi_reference_pipeline.constants import REF_TYPE_INTEGRALNONLINEARITY
from wfi_reference_pipeline.reference_types.integral_non_linearity.integral_non_linearity import (
    IntegralNonLinearity,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaIntegralNonLinearity metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_INTEGRALNONLINEARITY)
    return test_meta.meta_integral_non_linearity


@pytest.fixture
def inl_array():
    """Return a valid 2D INL correction array shape (32, 65536)."""
    return np.zeros((32, 65536), dtype=np.float64)


@pytest.fixture
def integral_non_linearity_object(valid_meta_data, inl_array):
    """Fixture for initializing an IntegralNonLinearity object with valid metadata."""
    return IntegralNonLinearity(meta_data=valid_meta_data, ref_type_data=inl_array)


class TestIntegralNonLinearity:

    def test_integral_non_linearity_instantiation_with_valid_metadata(
        self, integral_non_linearity_object
    ):
        """
        Test that IntegralNonLinearity object is created successfully with valid metadata.
        """
        assert isinstance(integral_non_linearity_object, IntegralNonLinearity)

    def test_default_description_is_set_if_empty(self, valid_meta_data, inl_array):
        """
        Test that a default description is populated if metadata description is empty.
        """
        valid_meta_data.description = ""
        inl_obj = IntegralNonLinearity(meta_data=valid_meta_data, ref_type_data=inl_array)

        assert inl_obj.meta_data.description == "Roman WFI integral non linearity reference file."

    def test_populate_datamodel_tree(self, integral_non_linearity_object):
        """
        Test that the data model tree is correctly populated.
        """
        data_model_tree = integral_non_linearity_object.populate_datamodel_tree()

        assert "meta" in data_model_tree
        assert "inl_table" in data_model_tree
        assert "value" in data_model_tree

        inl_table = data_model_tree["inl_table"]
        assert isinstance(inl_table, dict)
        assert len(inl_table) == 32

        # Ensure all keys exist and values are correct
        # There are 32 amplifiers to read 128 pixels at a time. 
        # https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/coordinate-systems
        for i in range(32):
            key = f"science_channel_{i+1:02d}"
            assert key in inl_table

            table_entry = inl_table[key]
            assert "instrument_channel" in table_entry
            assert "correction" in table_entry

            assert table_entry["instrument_channel"] == i
            assert isinstance(table_entry["correction"], np.ndarray)
            # The Analog to Digital conversion is in UINT16.
            assert table_entry["correction"].shape == (65536,)

        value_array = data_model_tree["value"]
        assert isinstance(value_array, np.ndarray)
        assert value_array.shape == (65536,)

    def test_integral_non_linearity_outfile_default(self, integral_non_linearity_object):
        """
        Test that the default outfile name is correct.
        """
        assert integral_non_linearity_object.outfile == "roman_inl.asdf"

    def test_invalid_ref_type_data_shape_raises_value_error(self, valid_meta_data):
        """
        Test that invalid INL array shapes raise a ValueError.
        """
        bad_array = np.zeros((10, 10))
        with pytest.raises(ValueError):
            IntegralNonLinearity(meta_data=valid_meta_data, ref_type_data=bad_array)

    def test_invalid_meta_type_raises_type_error(self, inl_array):
        """
        Test that invalid meta type raises a TypeError.
        """
        with pytest.raises(TypeError):
            IntegralNonLinearity(meta_data=object(), ref_type_data=inl_array)