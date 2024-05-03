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
    obj = Dark(test_meta.meta_dark, data_array=test_read_cube)
    obj.make_ma_table_resampled_cube(num_resultants=3, num_rds_per_res=1)
    yield obj


class TestDark:

    def test_dark_with_invalid_instatiate_fail(self, dark_object, read_cube):
        with pytest.raises(ValueError):
            Dark(dark_object.meta_data)

        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            Dark(bad_test_meta.meta_readnoise, data_array=read_cube)

        with pytest.raises(TypeError):
            Dark(dark_object.meta_data, data_array='not_data.txt')

    def test_dark_with_valid_data_array_pass(self, dark_object):
        assert isinstance(dark_object, Dark)
        assert dark_object.outfile == "roman_dark.asdf"
        assert dark_object.meta_data.description == "For RFP testing."

    def test_make_dark_rate_image_pass(self, dark_object):
        dark_object.make_dark_rate_image()
        assert dark_object.dark_rate_image.shape == (4096, 4096)
        assert dark_object.dark_rate_var.shape == (4096, 4096)
        assert dark_object.num_resultants == 3
        assert dark_object.ni == 4096
