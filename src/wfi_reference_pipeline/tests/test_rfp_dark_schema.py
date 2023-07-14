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

        # Make reftype specific data class object and export meta data as dict.
        tmp = MakeTestMeta(ref_type='DARK')
        dark_test_meta = tmp.meta_dark.export_asdf_meta()

        # Make RFP Dark reference file object for testing.
        test_data = np.ones((3, 3, 3), dtype=np.float32)
        rfp_dark = Dark('test_file.txt', meta_data=dark_test_meta)

        # Build dark reference asdf file object and test by asserting validate returns none.
        drk = rds.DarkRef()
        # TODO - do we want reference file to output data/err/dq in correct format for each of these?
        drk['meta'] = rfp_dark.meta
        drk['data'] = test_data * u.DN
        drk['err'] = test_data * u.DN
        drk['dq'] = rfp_dark.mask
        td = asdf.AsdfFile()
        td.tree = {'roman': drk}
        # The validate method will return a list of exceptions that the schema fails to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert td.validate() is None




