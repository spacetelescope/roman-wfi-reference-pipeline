import os

import numpy as np
import pytest

#from roman_datamodels import maker_utils as utils
from wfi_reference_pipeline.constants import REF_TYPE_READNOISE, REF_TYPE_REFPIX
from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import (
    ReferencePixel,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

@pytest.fixture(scope="module")
def valid_meta_data():
    """Fixture for generating valid meta_data for ReferencePixel class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_REFPIX)
    return test_meta.meta_referencepixel

@pytest.fixture(scope="module")
def valid_ref_type_data_cube():
    """Fixture for generating valid ref_type_data cube.
    """
    np.random.seed(42) # Fixed seed for reproducibility

    random_refpix_exposure = np.random.random((3, 4096, 4224))
    return random_refpix_exposure  # Simulate a valid refpix data cube

@pytest.fixture(scope="module")
def refpix_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a ReferencePixel object with two valid data cube."""
    refpix_object_with_data_cube = ReferencePixel(meta_data=valid_meta_data,
                                      ref_type_data=np.array(valid_ref_type_data_cube))
    return refpix_object_with_data_cube


class TestRefPix:
    def test_refpix_instantiation_with_valid_ref_type_data_cube(self, refpix_object_with_data_cube):
        """
        Test that RefPix object is created successfully with valid input data cube.
        """
        assert isinstance(refpix_object_with_data_cube, ReferencePixel)
        assert refpix_object_with_data_cube.ref_type_data is not None
        assert refpix_object_with_data_cube.ref_type_data.shape == (1, 3, 4096, 4224)

        assert refpix_object_with_data_cube.gamma is None  # Ensure gamma array is not created yet
        assert refpix_object_with_data_cube.zeta is None  # Ensure zeta array is not created yet
        assert refpix_object_with_data_cube.alpha is None  # Ensure alpha array is not created yet

    def test_refpix_instantiation_with_invalid_metadata(self, refpix_object_with_data_cube):
        """
        Test that RefPix raises TypeError with invalid metadata type.
        """
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            ReferencePixel(meta_data=bad_test_meta.meta_readnoise, ref_type_data=refpix_object_with_data_cube)

    def test_refpix_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """
        Test that RefPix raises TypeError with invalid reference type data.
        """
        with pytest.raises(TypeError):
            ReferencePixel(meta_data=valid_meta_data, ref_type_data='invalid_ref_data')


    def test_refpix_instantiation_with_file_list(self, valid_meta_data):
        """
        Test that RefPix object handles file list input correctly.
        """

        mock_file_list = ["file1.asdf", "file2.asdf"]
        refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)
        assert len(refpix_obj.file_list) == 2
        assert isinstance(refpix_obj.file_list, list)

    # @skip_on_github # TODO - fix this
    # def test_get_data_cube_from_file(self, tmp_path, valid_meta_data):
    #     """
    #     Test open data cube from input file list
    #     """

    #     file_path = tmp_path / "test.asdf"
    #     with asdf.AsdfFile() as af:
    #         af.tree = {"roman": utils.mk_level1_science_raw(shape=(5, 4096, 4096))}
    #         af.write_to(file_path)

    #     mock_file_list = [file_path]
    #     refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)
    #     print(refpix_obj.file_list)
    #     refpix_data = refpix_obj.get_data_cube_from_file(refpix_obj.file_list[0], skip_first_frame=False)

    #     assert refpix_data.shape == (5, 4096, 4224)
    #     assert refpix_data.dtype == np.float64

    # @skip_on_github # TODO - fix this
    # def test_get_detector_name_from_data_file_meta(self, tmp_path, valid_meta_data):
    #     """
    #     Test open data cube from input file list
    #     """
    #     # Create a mock for the file content with the expected structure
    #     file_path = tmp_path / "test.asdf"
    #     with asdf.AsdfFile() as af:
    #         af.tree = {"roman": utils.mk_level1_science_raw(shape=(1, 4096, 4096))}
    #         af.tree['roman']['meta']['instrument']['detector'] = 'WFI01'
    #         af.write_to(file_path)

    #     mock_file_list = [file_path]
    #     refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)
    #     refpix_obj._get_detector_name_from_data_file_meta(mock_file_list[0])

    #     assert refpix_obj.meta_data.instrument_detector == 'WFI01'


    def test_make_referencepixel_image(self, refpix_object_with_data_cube):
        # Assert that make_referencepixel_image was called with tmppath=None (default)
        refpix_object_with_data_cube.make_referencepixel_image(tmppath=None, detector_name='WFI01')

        # Check that data_rate_image and data_rate_image_error are set
        assert refpix_object_with_data_cube.gamma is not None
        assert refpix_object_with_data_cube.zeta is not None
        assert refpix_object_with_data_cube.alpha is not None

        assert refpix_object_with_data_cube.gamma.shape == (32, 286721)
        assert refpix_object_with_data_cube.gamma.dtype == np.complex128
        assert refpix_object_with_data_cube.zeta.shape == (32, 286721)
        assert refpix_object_with_data_cube.zeta.dtype == np.complex128
        assert refpix_object_with_data_cube.alpha.shape == (32, 286721)
        assert refpix_object_with_data_cube.alpha.dtype == np.complex128


    @skip_on_github
    def test_populate_datamodel_tree(self, refpix_object_with_data_cube):
        """
        Test that the data model tree is correctly populated in the RefPix object.
        """
        refpix_object_with_data_cube.gamma = np.zeros((32, 286721), dtype=complex)
        refpix_object_with_data_cube.zeta = np.zeros((32, 286721), dtype=complex)
        refpix_object_with_data_cube.alpha = np.zeros((32, 286721), dtype=complex)
        data_model_tree = refpix_object_with_data_cube.populate_datamodel_tree()

        # Assuming the RefPix data model includes:
        assert 'meta' in data_model_tree
        assert 'gamma' in data_model_tree
        assert 'zeta' in data_model_tree
        assert 'alpha' in data_model_tree

        # Check the shape and dtype of the 'gamma' array
        assert data_model_tree['gamma'].shape == (32, 286721)
        assert data_model_tree['gamma'].dtype == np.complex128
        # Check the shape and dtype of the 'zeta' array
        assert data_model_tree['zeta'].shape == (32, 286721)
        assert data_model_tree['zeta'].dtype == np.complex128
        # Check the shape and dtype of the 'alpha' array
        assert data_model_tree['alpha'].shape == (32, 286721)
        assert data_model_tree['alpha'].dtype == np.complex128


    def test_refpix_outfile_default(self, refpix_object_with_data_cube):
        """
        Test that the default outfile name is correct in the RefPix object with the assumption
        that the default name is 'roman_refpix.asdf'
        """
        assert refpix_object_with_data_cube.outfile == "roman_referencepixel.asdf"