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
def refpix_object_with_data_cube(valid_meta_data, valid_ref_type_data_cube):
    """Fixture for initializing a Flat object with a valid data cube."""
    refpix_object_with_data_cube = ReferencePixel(meta_data=valid_meta_data,
                                      ref_type_data=valid_ref_type_data_cube)
    yield refpix_object_with_data_cube



@pytest.fixture
def refpix_coefficients_3_by_3():
    """Fixture for a testable reference pixel coefficients.

    Returning array in top row that is above threshold values,
    middle row which is equal to threshold values, and bottom
    row that is below.

    array flags = [ hot, warm, good,
                    hot, warm, dead,
                    good, dead, dead]
    """
    return np.array([
        [2.1, 1.1, 0.2],  # should return hot, warm, no flag set
        [2.0, 1.0, 0.1],  # should return hot, warm, dead
        [0.5, 0.0, -0.1],  # should return no flag set, dead, dead
    ])

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


    def test_make_referencepixel_image(self, refpix_object_with_data_cube):
        mock_return_image = np.random.rand(5, 10, 10)
        refpix_object_with_data_cube.ref_type_data = mock_return_image

        # Assert that make_referencepixel_image was called with tmppath=None (default)
        refpix_object_with_data_cube.make_referencepixel_image.assert_called_once_with(tmppath=None)

        refpix_object_with_data_cube.make_referencepixel_image()

        # Check that dark_rate_image and dark_rate_image_error are set
        assert refpix_object_with_data_cube.gamma is not None
        assert refpix_object_with_data_cube.zeta is not None
        assert refpix_object_with_data_cube.alpha is not None

        # assert refpix_object_with_data_cube.gamma.shape == mock_return_image.shape
        # assert dark_object_with_data_cube.dark_rate_image_error.shape == mock_return_error_image.shape

    # @skip_on_github
    # def test_populate_datamodel_tree(self, refpix_object_with_data_cube,
    #                                  valid_ref_type_data_cube,
    #                                  dark_rate_image_3_by_3):
    #     """
    #     Test that the data model tree is correctly populated in the Dark object.
    #     """
    #     dark_object_with_data_cube.resampled_data = valid_ref_type_data_cube
    #     dark_object_with_data_cube.dark_rate_image = dark_rate_image_3_by_3
    #     dark_object_with_data_cube.dark_rate_image_error = dark_rate_image_3_by_3
    #     data_model_tree = dark_object_with_data_cube.populate_datamodel_tree()

    #     # Assuming the Flat data model includes:
    #     assert 'meta' in data_model_tree
    #     assert 'data' in data_model_tree
    #     assert 'dark_slope' in data_model_tree
    #     assert 'dark_slope_error' in data_model_tree
    #     assert 'dq' in data_model_tree

    #     # Check the shape and dtype of the 'data' array
    #     assert data_model_tree['data'].shape == (5, 4096, 4096)
    #     assert data_model_tree['data'].dtype == np.float32
    #     # Check the shape and dtype of the 'dark_slope' array
    #     assert data_model_tree['dark_slope'].shape == (3, 3)
    #     assert data_model_tree['dark_slope'].dtype == np.float32
    #     # Check the shape and dtype of the 'dark_slope_error' array
    #     assert data_model_tree['dark_slope_error'].shape == (3, 3)
    #     assert data_model_tree['dark_slope_error'].dtype == np.float32
    #     # Check the shape and dtype of the 'dq' array
    #     assert data_model_tree['dq'].shape == (4096, 4096)
    #     assert data_model_tree['dq'].dtype == np.uint32

    def test_refpix_outfile_default(self, refpix_object_with_data_cube):
        """
        Test that the default outfile name is correct in the RefPix object with the assumption
        that the default name is 'roman_refpix.asdf'
        """
        assert refpix_object_with_data_cube.outfile == "roman_refpix.asdf"