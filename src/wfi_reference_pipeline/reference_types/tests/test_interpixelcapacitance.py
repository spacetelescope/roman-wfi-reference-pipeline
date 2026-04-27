import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    REF_TYPE_INTERPIXELCAPACITANCE,
    REF_TYPE_GAIN,
)
from wfi_reference_pipeline.reference_types.inter_pixel_capacitance.inter_pixel_capacitance import InterPixelCapacitance
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaIPC metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_INTERPIXELCAPACITANCE)
    return test_meta.meta_ipc


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating a valid 3x3 IPC kernel."""
    return np.random.random((3, 3)).astype(np.float32)


@pytest.fixture
def ipc_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing IPC object with user-provided kernel."""
    ipc_object = InterPixelCapacitance(
        meta_data=valid_meta_data,
        ref_type_data=valid_ref_type_data_array
    )
    yield ipc_object


class TestIPC:

    def test_ipc_instantiation_with_valid_ref_type_data(self, ipc_object_with_data_array):
        """
        Test that IPC object is created successfully with valid input data.
        """
        assert isinstance(ipc_object_with_data_array, InterPixelCapacitance)
        assert ipc_object_with_data_array.ipc_kernel.shape == (3, 3)

    def test_ipc_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """
        Test that IPC raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_GAIN)

        with pytest.raises(TypeError):
            InterPixelCapacitance(
                meta_data=bad_test_meta.meta_gain,
                ref_type_data=valid_ref_type_data_array
            )

    def test_ipc_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that IPC raises errors with invalid reference type data.
        """
        with pytest.raises(TypeError):
            InterPixelCapacitance(
                meta_data=valid_meta_data,
                ref_type_data="invalid_ref_data"
            )

    def test_populate_datamodel_tree(self, ipc_object_with_data_array):
        """
        Test that the data model tree is correctly populated in the IPC object.
        """
        data_model_tree = ipc_object_with_data_array.populate_datamodel_tree()

        # Check required keys
        assert 'meta' in data_model_tree
        assert 'data' in data_model_tree

        # Check kernel shape and dtype
        assert data_model_tree['data'].shape == (3, 3)
        assert data_model_tree['data'].dtype == np.float32

    def test_ipc_outfile_default(self, ipc_object_with_data_array):
        """
        Test that the default outfile name is correct.
        """
        assert ipc_object_with_data_array.outfile == "roman_ipc.asdf"