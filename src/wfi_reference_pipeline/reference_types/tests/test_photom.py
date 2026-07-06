import pytest

from wfi_reference_pipeline.constants import REF_TYPE_EPSF, REF_TYPE_PHOTOM
from wfi_reference_pipeline.reference_types.photom.photom import (
    Photom,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.resources.wfi_meta_photom import (
    WFIMetaPhotom,
)


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaPhotom metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_PHOTOM)
    return test_meta.meta_photom



@pytest.fixture
def photom_object(valid_meta_data):
    """Fixture for initializing a Photom object with valid data."""
    photom_object = Photom(meta_data=valid_meta_data)  
    return photom_object


class TestPhotom:

    def test_photom_instantiation_with_valid_metadata(
        self,
        photom_object,
    ):
        """
        Test that Photom object is created successfully with valid metadata.
        """
        assert isinstance(photom_object, Photom)
        assert isinstance(photom_object.meta_data, WFIMetaPhotom)


    def test_photom_instantiation_with_invalid_metadata(
        self
    ):
        """
        Test that Photom raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_EPSF)

        with pytest.raises(TypeError):
            Photom(
                meta_data=bad_test_meta.meta_epsf,
            )


    def test_populate_datamodel_tree(
        self,
        photom_object,
    ):
        """
        Test that the datamodel tree is correctly populated.
        """
        data_model_tree = (
            photom_object.populate_datamodel_tree()
        )

        assert 'meta' in data_model_tree
        assert 'phot_table' in data_model_tree

        assert isinstance(data_model_tree['phot_table'], dict)

        keys = data_model_tree['meta']['instrument'].keys()
        assert 'detector' in keys
        assert 'median_gain' in keys
        assert 'sigma_gain' in keys


    def test_photom_outfile_default(
        self,
        photom_object,
    ):
        """
        Test that the default outfile name is correct.
        """
        assert (
            photom_object.outfile == 'roman_photom_file.asdf'
        )