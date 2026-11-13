import pytest

from wfi_reference_pipeline.constants import REF_TYPE_EPSF, REF_TYPE_ETC
from wfi_reference_pipeline.reference_types.exposure_time_calculator.exposure_time_calculator import (
    ExposureTimeCalculator,
)
from wfi_reference_pipeline.resources.make_meta import MakeMeta
from wfi_reference_pipeline.resources.wfi_meta_exposure_time_calculator import (
    WFIMetaETC,
)


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaETC metadata."""
    test_meta = MakeMeta(ref_type=REF_TYPE_ETC, instrument_detector="WFI01")
    return test_meta.meta_etc



@pytest.fixture
def etc_object(valid_meta_data):
    """Fixture for initializing an ETC object with valid data."""
    etc_object = ExposureTimeCalculator(meta_data=valid_meta_data)
    return etc_object


class TestETC:

    def test_etc_instantiation_with_valid_metadata(
        self,
        etc_object,
    ):
        """
        Test that ETC object is created successfully with valid metadata.
        """
        assert isinstance(etc_object, ExposureTimeCalculator)
        assert isinstance(etc_object.meta_data, WFIMetaETC)
        assert len(etc_object.etc_detector_form) == 28


    def test_etc_instantiation_with_invalid_metadata(
        self
    ):
        """
        Test that ETC raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeMeta(ref_type=REF_TYPE_EPSF, instrument_detector="WFI01")

        with pytest.raises(TypeError):
            ExposureTimeCalculator(
                meta_data=bad_test_meta.meta_epsf
            )


    def test_populate_datamodel_tree(
        self,
        etc_object,
    ):
        """
        Test that the datamodel tree is correctly populated.
        """
        data_model_tree = (
            etc_object.populate_datamodel_tree()
        )

        assert "meta" in data_model_tree
        assert "form" in data_model_tree

        assert isinstance(data_model_tree["form"], dict)

        keys = data_model_tree["form"].keys()
        assert "saturation_fullwell" in keys
        assert "dark_current" in keys
        assert "readnoise" in keys
        assert "flat_field_electrons" in keys
        assert data_model_tree["form"]["saturation_fullwell"] != 0
        assert data_model_tree["form"]["dark_current"] != 0
        assert data_model_tree["form"]["readnoise"] != 0
        assert data_model_tree["form"]["flat_field_electrons"] != 0


    def test_etc_outfile_default(
        self,
        etc_object,
    ):
        """
        Test that the default outfile name is correct.
        """
        assert (
            etc_object.outfile
            == "roman_etc_file.asdf"
        )