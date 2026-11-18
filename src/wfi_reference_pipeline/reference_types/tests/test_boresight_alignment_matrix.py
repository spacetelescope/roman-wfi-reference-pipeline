import numpy as np
import pytest
 
from wfi_reference_pipeline.constants import (
    REF_TYPE_BORESIGHTALIGNMENTMATRIX,
    REF_TYPE_DARKDECAYSIGNAL,
    WFI_DETECTOR_NOT_CONFIGURED,
)
from wfi_reference_pipeline.reference_types.boresight_alignment_matrix.boresight_alignment_matrix import (
    BAM_QUATERNION,
    BoresightAlignmentMatrix,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
 
 
@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaBoresightAlignmentMatrix metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_BORESIGHTALIGNMENTMATRIX)
    return test_meta.meta_bam
 
 
@pytest.fixture
def valid_ref_type_data_quaternion():
    """Fixture for generating a valid unit quaternion."""
    return list(BAM_QUATERNION)
 
 
@pytest.fixture
def bam_object_with_quaternion(valid_meta_data, valid_ref_type_data_quaternion):
    """Fixture for initializing a BoresightAlignmentMatrix object with a valid quaternion."""
    return BoresightAlignmentMatrix(meta_data=valid_meta_data,
                                    ref_type_data=valid_ref_type_data_quaternion)
 
 
class TestBoresightAlignmentMatrix:
 
    def test_bam_instantiation_with_valid_ref_type_data(self, bam_object_with_quaternion):
        """
        Test that BoresightAlignmentMatrix object is created successfully with a valid quaternion.
        """
        assert isinstance(bam_object_with_quaternion, BoresightAlignmentMatrix)
        assert bam_object_with_quaternion.bam_quaternion.shape == (4,)
 
    def test_bam_instantiation_without_ref_type_data_uses_default(self, valid_meta_data):
        """
        Test that BAM requires no input data and falls back to BAM_QUATERNION.
        """
        rfp_bam = BoresightAlignmentMatrix(meta_data=valid_meta_data)
        np.testing.assert_allclose(rfp_bam.bam_quaternion, BAM_QUATERNION)
 
    def test_bam_instantiation_with_invalid_metadata(self, valid_ref_type_data_quaternion):
        """
        Test that metadata of a different reference type raises TypeError.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARKDECAYSIGNAL)
        with pytest.raises(TypeError):
            BoresightAlignmentMatrix(meta_data=bad_test_meta.meta_dark_decay_signal,
                                     ref_type_data=valid_ref_type_data_quaternion)
 
    def test_bam_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that invalid reference type data raises TypeError.
        """
        with pytest.raises(TypeError):
            BoresightAlignmentMatrix(meta_data=valid_meta_data,
                                     ref_type_data="invalid_ref_data")
 
    def test_bam_instantiation_with_wrong_quaternion_shape(self, valid_meta_data):
        """
        Test that a quaternion without exactly 4 elements raises ValueError.
        """
        with pytest.raises(ValueError):
            BoresightAlignmentMatrix(meta_data=valid_meta_data,
                                     ref_type_data=[0.0, 0.0, 1.0])
 
    def test_default_description_is_set_if_empty(self, valid_meta_data):
        """
        Test that a default description is populated if metadata description is empty.
        """
        valid_meta_data.description = ""
        rfp_bam = BoresightAlignmentMatrix(meta_data=valid_meta_data)
 
        assert rfp_bam.meta_data.description == (
            "Roman WFI Boresight Alignment Matrix (BAM) reference file."
        )
 
    def test_populate_datamodel_tree(self, bam_object_with_quaternion):
        """
        Test that the data model tree is correctly populated.
        """
        data_model_tree = bam_object_with_quaternion.populate_datamodel_tree()
 
        assert "meta" in data_model_tree
        assert "bam_quaternion" in data_model_tree
 
        meta = data_model_tree["meta"]
        assert meta["reftype"] == REF_TYPE_BORESIGHTALIGNMENTMATRIX
        assert meta["telescope"] == "ROMAN"
        assert meta["instrument"]["detector"] == WFI_DETECTOR_NOT_CONFIGURED
        assert {"v2ref", "v3ref", "v3ideal_y_angle"}.issubset(meta.keys())
 
        bam_quaternion = data_model_tree["bam_quaternion"]
        assert len(bam_quaternion) == 4
        assert all(isinstance(q, float) for q in bam_quaternion)
 
    def test_bam_quaternion_structure(self):
        """
        Test that the default BAM_QUATERNION has 4 elements and unit norm.
        """
        assert len(BAM_QUATERNION) == 4
        assert np.isclose(np.linalg.norm(BAM_QUATERNION), 1.0, atol=1.0e-6)
 
    def test_bam_outfile_default(self, bam_object_with_quaternion):
        """
        Test that the default outfile name is correct.
        """
        assert bam_object_with_quaternion.outfile == "roman_bam.asdf"
        