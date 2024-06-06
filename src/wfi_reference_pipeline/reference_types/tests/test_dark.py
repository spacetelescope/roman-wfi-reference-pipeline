import pytest
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_DARK, REF_TYPE_READNOISE
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads



@pytest.fixture(scope="class")
def read_cube():
    read_cube, _ = simulate_dark_reads(10)
    yield read_cube

@pytest.fixture(scope="class")
def dark_object():
    test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
    test_read_cube, _ = simulate_dark_reads(3)
    obj = Dark(test_meta.meta_dark, ref_type_data=test_read_cube)
    obj.make_rate_image_from_data_cube()
    obj.make_ma_table_resampled_data(num_resultants=3, num_reads_per_resultant=1)
    yield obj


class TestDark:

    def test_dark_with_invalid_instatiate_fail(self, dark_object, read_cube):
        with pytest.raises(ValueError):
            Dark(dark_object.meta_data)

        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Dark(bad_test_meta.meta_readnoise, ref_type_data=read_cube)

        with pytest.raises(TypeError):
            Dark(dark_object.meta_data, ref_type_data='not_data.txt')

    def test_dark_with_valid_data_array_pass(self, dark_object):
        assert isinstance(dark_object, Dark)
        assert dark_object.outfile == "roman_dark.asdf"
        assert dark_object.meta_data.description == "For RFP testing."

    def test_dark_get_rate_image_pass(self, dark_object):
        assert dark_object.data_cube.rate_image.shape == (4096, 4096)
        assert dark_object.num_resultants == 3
        assert dark_object.data_cube.num_i_pixels == 4096
        assert dark_object.data_cube.num_j_pixels == 4096
