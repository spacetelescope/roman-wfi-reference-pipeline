import numpy as np
import pytest

from wfi_reference_pipeline.constants import REF_TYPE_EPSF, REF_TYPE_GAIN
from wfi_reference_pipeline.reference_types.empirical_psf.empirical_psf import (
    EmpiricalPSF,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.resources.wfi_meta_empirical_psf import (
    WFIMetaEPSF,
)


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaEPSF metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_EPSF)
    return test_meta.meta_epsf


@pytest.fixture
def valid_psf_data():
    """Fixture for generating valid PSF reference data."""
    return np.random.random((10, 25, 25)).astype(np.float32)


@pytest.fixture
def valid_extended_psf_data():
    """Fixture for generating valid extended PSF reference data."""
    return np.random.random((10, 51, 51)).astype(np.float32)


@pytest.fixture
def empirical_psf_object(valid_meta_data, valid_psf_data):
    """Fixture for initializing an EmpiricalPSF object with valid data."""
    empirical_psf_object = EmpiricalPSF(
        meta_data=valid_meta_data,
        psf=valid_psf_data,
    )
    yield empirical_psf_object


class TestEmpiricalPSF:

    def test_empirical_psf_instantiation_with_valid_data(
        self,
        empirical_psf_object,
    ):
        """
        Test that EmpiricalPSF object is created successfully with valid input data.
        """
        assert isinstance(empirical_psf_object, EmpiricalPSF)
        assert isinstance(empirical_psf_object.meta, WFIMetaEPSF)
        assert empirical_psf_object.psf.shape == (10, 25, 25)

    def test_empirical_psf_instantiation_with_invalid_metadata(
        self,
        valid_psf_data,
    ):
        """
        Test that EmpiricalPSF raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_GAIN)

        with pytest.raises(TypeError):
            EmpiricalPSF(
                meta_data=bad_test_meta.meta_gain,
                psf=valid_psf_data,
            )

    def test_empirical_psf_instantiation_with_invalid_psf_data(
        self,
        valid_meta_data,
    ):
        """
        Test that EmpiricalPSF raises TypeError with invalid PSF data type.
        """
        with pytest.raises(TypeError):
            EmpiricalPSF(
                meta_data=valid_meta_data,
                psf="invalid_psf_data",
            )

    def test_update_meta_data_converts_fields_to_lists(
        self,
        empirical_psf_object,
    ):
        """
        Test that metadata iterable fields are converted to lists.
        """
        assert isinstance(empirical_psf_object.meta.pixel_x, list)
        assert isinstance(empirical_psf_object.meta.pixel_y, list)
        assert isinstance(empirical_psf_object.meta.spectral_type, list)
        assert isinstance(empirical_psf_object.meta.defocus, list)

    def test_populate_datamodel_tree(
        self,
        empirical_psf_object,
    ):
        """
        Test that the datamodel tree is correctly populated.
        """
        data_model_tree = (
            empirical_psf_object.populate_datamodel_tree()
        )

        assert "meta" in data_model_tree
        assert "psf" in data_model_tree

        assert data_model_tree["psf"].shape == (10, 25, 25)
        assert data_model_tree["psf"].dtype == np.float32

    def test_populate_datamodel_tree_with_optional_arrays(
        self,
        valid_meta_data,
        valid_psf_data,
        valid_extended_psf_data,
    ):
        """
        Test that optional PSF arrays are added to the datamodel tree.
        """
        empirical_psf_object = EmpiricalPSF(
            meta_data=valid_meta_data,
            psf=valid_psf_data,
            extended_psf=valid_extended_psf_data,
            psf_noipc=valid_psf_data,
            extended_psf_noipc=valid_extended_psf_data,
        )

        data_model_tree = (
            empirical_psf_object.populate_datamodel_tree()
        )

        assert "extended_psf" in data_model_tree
        assert "psf_noipc" in data_model_tree
        assert "extended_psf_noipc" in data_model_tree

        assert (
            data_model_tree["extended_psf"].shape
            == (10, 51, 51)
        )

    def test_empirical_psf_outfile_default(
        self,
        empirical_psf_object,
    ):
        """
        Test that the default outfile name is correct.
        """
        assert (
            empirical_psf_object.outfile
            == "roman_epsf_file.asdf"
        )
        