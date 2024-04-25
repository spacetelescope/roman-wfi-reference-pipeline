import numpy as np
import pytest
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_READNOISE, REF_TYPE_DARK
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
test_read_cube, test_rate_image = simulate_dark_reads(10)


class TestReadNoise():

    def test_readnoise_with_invalid_instatiate_fail(self):
        with pytest.raises(ValueError):
            ReadNoise(test_meta.meta_readnoise)

        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_DARK)
        with pytest.raises(TypeError):
            ReadNoise(bad_test_meta.meta_dark, data_array=test_read_cube)

        with pytest.raises(ValueError):
            ReadNoise(test_meta.meta_readnoise, data_array=np.zeros(10))


    def test_readnoise_with_valid_data_array_pass(self):
        test_meta.meta_readnoise.description = ""
        self.test_readnoise = ReadNoise(test_meta.meta_readnoise, data_array=test_read_cube)
        assert isinstance(self.test_readnoise, ReadNoise)
        assert self.test_readnoise.outfile == "roman_readnoise.asdf"
        assert self.test_readnoise.meta_data.description == "Roman WFI read noise reference file."




