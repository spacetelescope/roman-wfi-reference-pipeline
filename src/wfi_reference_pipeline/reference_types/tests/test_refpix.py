import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from romancal.lib import dqflags

from wfi_reference_pipeline.constants import REF_TYPE_REFPIX, REF_TYPE_READNOISE
from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import ReferencePixel
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for ReferencePixel class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_REFPIX)
    return test_meta.meta_referencepixel

@pytest.fixture
def valid_ref_type_data_cube():
    """Fixture for generating valid ref_type_data cube (reference pixel dark cube).
    """
    # from test data, assume a mean and std distribution on a per frame level of 4200 DN with std of 800 DN. Must have a 4224 by 4096 pixel frame (4096+128 rows to include amp 33).  
    random_refpix_frame = np.random.normal(4200, 800, 4224*4096) 
    # up the ramp, assume a std of 4.5 DN per pixel.  
    random_refpix_exp = np.array([np.random.normal(random_refpix_frame, np.ones_like(random_refpix_frame)*4.5, 4224*4096) for i in np.arange(5)]).reshape((5,4224, 4096)).astype(int)

    return random_refpix_exp  # Simulate a valid refpix data cube



@pytest.fixture
def refpix_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a ReferencePixel object with a valid data cube."""
    refpix_object_with_data_cube = ReferencePixel(meta_data=valid_meta_data,
                                      ref_type_data=valid_ref_type_data_cube)
    yield refpix_object_with_data_cube

class TestRefPix:

    def test_refpix_instantiation_with_valid_ref_type_data_cube(self, refpix_object_with_data_cube):
        """
        Test that RefPix object is created successfully with valid input data cube.
        """
        assert isinstance(refpix_object_with_data_cube, ReferencePixel)
        assert refpix_object_with_data_cube.ref_type_data is not None
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

    @patch("asdf.open")
    def test_refpix_instantiation_with_file_list(self, mock_asdf_open, valid_meta_data):
        """
        Test that RefPix object handles file list input correctly.
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "roman": {
                "data": np.zeros((5, 10, 10))  # Mocking data
            }
        }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits", "file2.fits"]
        refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)

        assert refpix_obj.num_files == 2

    def test_get_data_cube_from_dark_file(self, valid_meta_data):
        """
        Test open data cube from input file list
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "roman": {
                "data": np.zeros((5, 4096, 4096)),  # Mocking data
                "amp33": np.zeros((5, 128, 4096))
            }
        }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits"]
        refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)
        refpix_data = refpix_obj.get_data_cube_from_dark_file(mock_file_list[0], skip_first_frame=False)

        assert refpix_data.shape == (5, 4224, 4096)
        assert refpix_data.dtype == np.float64

    def test_get_detector_name_from_dark_file_meta(self, valid_meta_data):
        """
        Test open data cube from input file list
        """
        # Create a mock for the file content with the expected structure
        mock_asdf_file = MagicMock()
        mock_asdf_file.tree = {
            "meta":{"instrument":{"detector":'WFI01'}} }

        # Set the mock to return this structure when asdf.open is called
        mock_asdf_open.return_value.__enter__.return_value = mock_asdf_file

        mock_file_list = ["file1.fits"]
        refpix_obj = ReferencePixel(meta_data=valid_meta_data, file_list=mock_file_list)
        refpix_obj._get_detector_name_from_dark_file_meta(mock_file_list[0])

        assert refpix_obj.meta_data.instrument_detector == 'WFI01'


    def test_make_referencepixel_image(self, refpix_object_with_data_cube):
        mock_return_image = valid_ref_type_data_cube()
        # make 2 exposures
        refpix_object_with_data_cube.ref_type_data = [mock_return_image,mock_return_image]

        # Assert that make_referencepixel_image was called with tmppath=None (default)
        refpix_object_with_data_cube.make_referencepixel_image.assert_called_once_with(tmppath=None)

        refpix_object_with_data_cube.make_referencepixel_image()

        # Check that dark_rate_image and dark_rate_image_error are set
        assert refpix_object_with_data_cube.gamma is not None
        assert refpix_object_with_data_cube.zeta is not None
        assert refpix_object_with_data_cube.alpha is not None

        assert refpix_object_with_data_cube.gamma.shape == (33, 286721)
        assert refpix_object_with_data_cube.zeta.shape == (33, 286721)
        assert refpix_object_with_data_cube.alpha.shape == (33, 286721)
    

    # @skip_on_github
    def test_populate_datamodel_tree(self, refpix_object_with_data_cube,
                                     valid_ref_type_data_cube,
                                     dark_rate_image_3_by_3):
        """
        Test that the data model tree is correctly populated in the Dark object.
        """
        refpix_object_with_data_cube.gamma = np.zeros((33, 286721), dtype=complex)
        refpix_object_with_data_cube.zeta = np.zeros((33, 286721), dtype=complex)
        refpix_object_with_data_cube.alpha = np.zeros((33, 286721), dtype=complex)
        data_model_tree = refpix_object_with_data_cube.populate_datamodel_tree()

        # Assuming the RefPix data model includes:
        assert 'meta' in data_model_tree
        assert 'gamma' in data_model_tree
        assert 'zeta' in data_model_tree
        assert 'alpha' in data_model_tree

        # Check the shape and dtype of the 'gamma' array
        assert data_model_tree['gamma'].shape == (33, 286721)
        assert data_model_tree['gamma'].dtype == np.complex128
        # Check the shape and dtype of the 'zeta' array
        assert data_model_tree['zeta'].shape == (33, 286721)
        assert data_model_tree['zeta'].dtype == np.complex128
        # Check the shape and dtype of the 'alpha' array
        assert data_model_tree['alpha'].shape == (33, 286721)
        assert data_model_tree['alpha'].dtype == np.complex128


    def test_refpix_outfile_default(self, refpix_object_with_data_cube):
        """
        Test that the default outfile name is correct in the RefPix object with the assumption
        that the default name is 'roman_refpix.asdf'
        """
        assert refpix_object_with_data_cube.outfile == "roman_refpix.asdf"