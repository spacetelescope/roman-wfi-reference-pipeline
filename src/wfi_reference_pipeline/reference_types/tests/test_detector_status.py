import pytest

from wfi_reference_pipeline.constants import REF_TYPE_DETECTORSTATUS
from wfi_reference_pipeline.reference_types.detector_status.detector_status import (
    DetectorStatus,
)
from wfi_reference_pipeline.resources.make_meta import MakeMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaDetectorStatus metadata."""
    test_meta = MakeMeta(ref_type=REF_TYPE_DETECTORSTATUS, instrument_detector="WFI01")
    return test_meta.meta_detector_status


@pytest.fixture
def detector_status_object(valid_meta_data):
    """Fixture for initializing a DetectorStatus object with valid metadata."""
    detector_status_object = DetectorStatus(meta_data=valid_meta_data)
    yield detector_status_object


class TestDetectorStatus:

    def test_detector_status_instantiation_with_valid_metadata(self, detector_status_object):
        """
        Test that DetectorStatus object is created successfully with valid metadata.
        """
        assert isinstance(detector_status_object, DetectorStatus)
        assert detector_status_object.status_info_dict is not None
        assert isinstance(detector_status_object.status_info_dict, dict)

    def test_detector_status_instantiation_with_invalid_metadata(self):
        """
        Test that DetectorStatus raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeMeta(ref_type="DARKDECAYSIGNAL", instrument_detector="WFI01")  # wrong ref type but a ref type that also does not require input data
        with pytest.raises(TypeError):
            DetectorStatus(meta_data=bad_test_meta.meta_dark_decay_signal)

    def test_default_description_is_set_if_empty(self, valid_meta_data):
        """
        Test that a default description is populated if metadata description is empty.
        """
        valid_meta_data.description = ""
        detector_status = DetectorStatus(meta_data=valid_meta_data)

        assert detector_status.meta_data.description == (
            "Roman WFI detector status reference file."
        )

    def test_populate_datamodel_tree(self, detector_status_object):
        """
        Test that the data model tree is correctly populated.
        """
        data_model_tree = detector_status_object.populate_datamodel_tree()

        # Verify expected keys exist
        assert "meta" in data_model_tree
        assert "status_info" in data_model_tree

        # Validate status_info structure
        status_info = data_model_tree["status_info"]
        assert isinstance(status_info, dict)
        assert len(status_info) == 18

        for key, value in status_info.items():
            assert key.startswith("WFI")
            assert "enabled" in value
            assert isinstance(value["enabled"], bool)

    def test_detector_status_outfile_default(self, detector_status_object):
        """
        Test that the default outfile name is correct.
        """
        assert detector_status_object.outfile == "roman_detector_status.asdf"
