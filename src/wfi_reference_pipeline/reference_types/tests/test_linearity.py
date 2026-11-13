import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_LINEARITY,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.linearity.linearity import Linearity
from wfi_reference_pipeline.resources.make_meta import MakeMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaLinearity metadata."""
    test_meta = MakeMeta(ref_type=REF_TYPE_LINEARITY, instrument_detector="WFI01")
    return test_meta.meta_linearity


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid linearity coefficient data."""
    return np.random.random(
        (
            5,
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )
    )


@pytest.fixture
def linearity_object_with_data_array(
    valid_meta_data,
    valid_ref_type_data_array,
):
    """Fixture for initializing a Linearity object with valid data."""
    linearity_object = Linearity(
        meta_data=valid_meta_data,
        ref_type_data=valid_ref_type_data_array,
    )
    yield linearity_object


class TestLinearity:

    def test_linearity_instantiation_with_valid_ref_type_data(
        self,
        linearity_object_with_data_array,
    ):
        """
        Test that Linearity object is created successfully with valid input data.
        """
        assert isinstance(linearity_object_with_data_array, Linearity)

        assert (
            linearity_object_with_data_array.lin_coeffs_array.shape
            == (
                5,
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

    def test_linearity_instantiation_with_invalid_metadata(
        self,
        valid_ref_type_data_array,
    ):
        """
        Test that Linearity raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeMeta(ref_type=REF_TYPE_READNOISE, instrument_detector="WFI01")

        with pytest.raises(TypeError):
            Linearity(
                meta_data=bad_test_meta.meta_readnoise,
                ref_type_data=valid_ref_type_data_array,
            )

    def test_linearity_instantiation_with_invalid_ref_type_data(
        self,
        valid_meta_data,
    ):
        """
        Test that Linearity raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            Linearity(
                meta_data=valid_meta_data,
                ref_type_data="invalid_ref_data",
            )

    def test_linearity_instantiation_with_invalid_dimensions(
        self,
        valid_meta_data,
    ):
        """
        Test that Linearity raises ValueError when input data
        is not a 3D numpy array.
        """
        invalid_array = np.random.random(
            (
                DETECTOR_PIXEL_X_COUNT,
                DETECTOR_PIXEL_Y_COUNT,
            )
        )

        with pytest.raises(ValueError):
            Linearity(
                meta_data=valid_meta_data,
                ref_type_data=invalid_array,
            )

    def test_populate_datamodel_tree(
        self,
        linearity_object_with_data_array,
    ):
        """
        Test that the data model tree is correctly populated
        in the Linearity object.
        """
        data_model_tree = (
            linearity_object_with_data_array.populate_datamodel_tree()
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

    def test_linearity_outfile_default(
        self,
        linearity_object_with_data_array,
    ):
        """
        Test that the default outfile name is correct.
        """
        assert (
            linearity_object_with_data_array.outfile
            == "roman_linearity.asdf"
        )