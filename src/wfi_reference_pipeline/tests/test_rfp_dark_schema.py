import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.dark.dark import Dark
    from astropy import units as u


    def test_rfp_dark_schema():
        """
        Use the WFI reference file pipeline Dark() module to build a testable object which is then validated against
        the DMS reference file schema.
        """
        dark_test_meta = MakeTestMeta(ref_type='DARK')  # Make reftype specific data class object
        dark_meta = dark_test_meta.meta_dark.export_asdf_meta()  # export object as dict

        # Make RFP Dark reference file object for schema testing.
        test_data = np.ones((3, 3, 3), dtype=np.float32)
        rfp_dark = Dark('test_file.txt', dark_meta)

        # Build dark reference asdf file object and test by asserting validate returns none.
        rfp_test_dark = rds.DarkRef()
        # TODO - do we want reference file to output data/err/dq in correct format for each of these?
        rfp_test_dark['data'] = test_data * u.DN
        rfp_test_dark['err'] = test_data * u.DN
        rfp_test_dark['dq'] = rfp_dark.mask
        rfp_test_dark['meta'] = rfp_dark.meta
        td = asdf.AsdfFile()
        td.tree = {'roman': rfp_test_dark}
        # The validate method will return a list of exceptions that the schema fails to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert td.validate() is None




