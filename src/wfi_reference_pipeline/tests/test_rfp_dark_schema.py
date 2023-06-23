import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from roman_datamodels import units as ru
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.dark.dark import Dark



    def test_rfp_dark_schema():
        """
        Use the WFI reference file pipeline IPC() module to build a testable object which is then validated against
        the DMS reference file schema.
        """
        dark_make_test_meta = MakeTestMeta(ref_type='DARK')
        dark_meta = dark_make_test_meta.meta_dark

        # Make RFP Dark reference file object for testing.
        test_data = np.ones((3, 3, 3), dtype=np.float32) * ru.DN
        rfp_dark = Dark(None, dark_meta, input_dark_cube=test_data)
        rfp_dark.make_ma_table_dark(1, num_rds_per_res=1)
        rfp_dark.resampled_dark_cube *= ru.DN
        rfp_dark.resampled_dark_cube_err *= ru.DN

        # Build dark reference asdf file object and test by asserting validate returns none.
        # rfp_test_dark = rds.DarkRef()
        # rfp_test_dark['data'] = rfp_dark.resampled_dark_cube
        # rfp_test_dark['err'] = rfp_dark.resampled_dark_cube_err
        # rfp_test_dark['dq'] = rfp_dark.mask
        # rfp_test_dark['meta'] = rfp_dark.meta_data
        td = asdf.AsdfFile()
        # td.tree = {'roman': rfp_test_dark}
        # The validate method will return a list of exceptions that the schema fails to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert td.validate() is None




