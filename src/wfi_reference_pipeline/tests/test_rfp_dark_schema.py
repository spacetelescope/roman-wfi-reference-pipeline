import os
import sys
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ()

# TODO Enable this test once RTB-DATABASE is up and running.
if not ON_GITLAB_ACTIONS:
    sys.exit(0)


import asdf
import numpy as np
import roman_datamodels.stnode as rds
from roman_datamodels import units as ru
from wfi_reference_pipeline.tests import make_test_meta
from wfi_reference_pipeline.dark import dark



def test_rfp_dark_schema():
    """
    Use the WFI reference file pipeline IPC() module to build a testable object which is then validated against
    the DMS reference file schema.
    """
    dark_test = make_test_meta.MakeTestMeta(ref_type='DARK')

    if dark_test.test_meta is None:
        raise ValueError(f'No meta to test.')
    else:
        # Make RFP Dark reference file object for testing.
        test_data = np.ones((3, 3, 3), dtype=np.float32) * ru.DN
        rfp_dark = dark.Dark(None, meta_data=dark_test.test_meta, input_dark_cube=test_data)
        rfp_dark.make_ma_table_dark(1, num_rds_per_res=1)
        rfp_dark.resampled_dark_cube *= ru.DN
        rfp_dark.resampled_dark_cube_err *= ru.DN

        # Build dark reference asdf file object and test by asserting validate returns none.
        rfp_test_dark = rds.DarkRef()
        rfp_test_dark['data'] = rfp_dark.resampled_dark_cube
        rfp_test_dark['err'] = rfp_dark.resampled_dark_cube_err
        rfp_test_dark['dq'] = rfp_dark.mask
        rfp_test_dark['meta'] = rfp_dark.meta
        td = asdf.AsdfFile()
        td.tree = {'roman': rfp_test_dark}
        # The validate method will return a list of exceptions that the schema fails to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert td.validate() is None




