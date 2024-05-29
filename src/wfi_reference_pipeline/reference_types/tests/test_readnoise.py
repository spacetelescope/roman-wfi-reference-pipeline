import pytest
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_READNOISE, REF_TYPE_DARK
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads



@pytest.fixture(scope="class")
def read_cube():
    read_cube, _ = simulate_dark_reads(10)
    yield read_cube

@pytest.fixture(scope="class")
def readnoise_object():
    test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
    test_read_cube, _ = simulate_dark_reads(3)
    obj = ReadNoise(test_meta.meta_readnoise, ref_type_data=test_read_cube)
    yield obj


class TestReadNoise:

    def test_readnoise_with_invalid_instatiate_fail(self, readnoise_object, read_cube):
        with pytest.raises(ValueError):
            ReadNoise(readnoise_object.meta_data)

        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
        with pytest.raises(TypeError):
            ReadNoise(bad_test_meta.meta_dark, ref_type_data=read_cube)

        with pytest.raises(TypeError):
            ReadNoise(readnoise_object.meta_data, ref_type_data='not_data.txt')

    def test_readnoise_with_valid_ref_type_data_pass(self, readnoise_object):
        assert isinstance(readnoise_object, ReadNoise)
        assert readnoise_object.outfile == "roman_readnoise.asdf"
        assert readnoise_object.meta_data.description == "For RFP testing."

    def test_make_readnoise_image_pass(self, readnoise_object):
        readnoise_object.make_readnoise_image()
        assert readnoise_object.readnoise_image.shape == (4096, 4096)
        assert readnoise_object.n_reads == 3
        assert readnoise_object.ni == 4096

