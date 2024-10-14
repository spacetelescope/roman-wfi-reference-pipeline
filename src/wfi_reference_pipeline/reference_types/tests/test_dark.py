import pytest
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_DARK, REF_TYPE_READNOISE
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid meta_data for Dark class."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
    return test_meta.meta_dark


@pytest.fixture
def valid_ref_type_data():
    """Fixture for generating valid ref_type_data (read cube)."""
    ref_type_data, _ = simulate_dark_reads(5)  # Simulate a 5-read data cube
    return ref_type_data


@pytest.fixture
def dark_object(valid_meta_data, valid_ref_type_data):
    """Fixture for initializing a valid Dark object."""
    dark_obj = Dark(meta_data=valid_meta_data, ref_type_data=valid_ref_type_data)
    dark_obj.make_rate_image_from_data_cube()
    dark_obj.make_ma_table_resampled_data(num_resultants=3, num_reads_per_resultant=1)
    yield dark_obj


class TestDark:

    def test_dark_instantiation_with_valid_data(self, dark_object):
        """
        Test that Dark object is created successfully with valid input data.
        """
        assert isinstance(dark_object, Dark)
        assert dark_object.data_cube.data.shape == (5, 4096, 4096)


    

    # def test_dark_with_invalid_instatiate_fail(self, dark_object, read_cube):
    #     with pytest.raises(ValueError):
    #         Dark(dark_object.meta_data)
    #
    #     bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
    #     with pytest.raises(TypeError):
    #         Dark(bad_test_meta.meta_readnoise, ref_type_data=read_cube)
    #
    #     with pytest.raises(TypeError):
    #         Dark(dark_object.meta_data, ref_type_data='not_data.txt')
    #
    # def test_dark_with_valid_ref_type_data_pass(self, dark_object):
    #     assert isinstance(dark_object, Dark)
    #     assert dark_object.outfile == "roman_dark.asdf"
    #     assert dark_object.meta_data.description == "For RFP testing."
    #
    # def test_dark_get_rate_image_pass(self, dark_object):
    #     assert dark_object.data_cube.rate_image.shape == (4096, 4096)
    #     assert dark_object.num_resultants == 3
    #     assert dark_object.data_cube.num_i_pixels == 4096
    #     assert dark_object.data_cube.num_j_pixels == 4096
