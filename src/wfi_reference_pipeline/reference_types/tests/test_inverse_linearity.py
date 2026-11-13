import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_INVERSELINEARITY,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.inverse_linearity.inverse_linearity import (
    InverseLinearity,
)
from wfi_reference_pipeline.resources.make_meta import MakeMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaInverseLinearity metadata."""
    test_meta = MakeMeta(ref_type=REF_TYPE_INVERSELINEARITY, instrument_detector="WFI01")
    return test_meta.meta_inverselinearity


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid inverse linearity coefficient data."""
    return np.random.random(
        (
            5,
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )
    )


@pytest.fixture
def inverse_linearity_object_with_data_array(
    valid_meta_data,
    valid_ref_type_data_array,
):
    """Fixture for initializing an InverseLinearity object with valid data."""
    inverse_linearity_object = InverseLinearity(
        meta_data=valid_meta_data,
        ref_type_data=valid_ref_type_data_array,
    )
    yield inverse_linearity_object


class TestInverseLinearity:

    def test_inverse_linearity_instantiation_with_valid_ref_type_data(
        self,
        inverse_linearity_object_with_data_array,
    ):
        """
        Test that InverseLinearity object is created successfully
        with valid input data.
        """
        assert isinstance(
            inverse_linearity_object_with_data_array,
            InverseLinearity,
        )

        assert (
            inverse_linearity_object_with_data_array.inverse_lin_coeffs_array.shape
            == (
                5,
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

    def test_inverse_linearity_instantiation_with_invalid_metadata(
        self,
        valid_ref_type_data_array,
    ):
        """
        Test that InverseLinearity raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeMeta(ref_type=REF_TYPE_READNOISE, instrument_detector="WFI01")

        with pytest.raises(TypeError):
            InverseLinearity(
                meta_data=bad_test_meta.meta_readnoise,
                ref_type_data=valid_ref_type_data_array,
            )

    def test_inverse_linearity_instantiation_with_invalid_ref_type_data(
        self,
        valid_meta_data,
    ):
        """
        Test that InverseLinearity raises TypeError with invalid
        reference type data.
        """
        with pytest.raises(TypeError):
            InverseLinearity(
                meta_data=valid_meta_data,
                ref_type_data="invalid_ref_data",
            )

    def test_inverse_linearity_instantiation_with_invalid_dimensions(
        self,
        valid_meta_data,
    ):
        """
        Test that InverseLinearity raises ValueError when input data
        is not a 3D numpy array.
        """
        invalid_array = np.random.random(
            (
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

        with pytest.raises(ValueError):
            InverseLinearity(
                meta_data=valid_meta_data,
                ref_type_data=invalid_array,
            )

    def test_populate_datamodel_tree(
        self,
        inverse_linearity_object_with_data_array,
    ):
        """
        Test that the data model tree is correctly populated
        in the InverseLinearity object.
        """
        data_model_tree = (
            inverse_linearity_object_with_data_array.populate_datamodel_tree()
        )

        assert "meta" in data_model_tree
        assert "coeffs" in data_model_tree
        assert "dq" in data_model_tree

        assert data_model_tree["coeffs"].shape == (
            5,
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )

        assert data_model_tree["coeffs"].dtype == np.float32

    def test_inverse_linearity_outfile_default(
        self,
        inverse_linearity_object_with_data_array,
    ):
        """
        Test that the default outfile name is correct.
        """
        assert (
            inverse_linearity_object_with_data_array.outfile
            == "roman_inverse_linearity.asdf"
        )