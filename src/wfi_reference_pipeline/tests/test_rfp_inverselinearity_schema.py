import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.inverselinearity.inverselinearity import InverseLinearity


    def test_rfp_inv_linear_schema():
        """
        Use the WFI reference file pipeline InvLinearity() module to build a testable object which is then validated against
        the DMS reference file schema.
        """
        invlin_test = MakeTestMeta(ref_type='INVERSELINEARITY')
        invlin_meta = invlin_test.meta_inverselinearity.export_asdf_meta()  # export object as dict

        # Make RFP Inverse Linearity reference file object for testing.
        test_data = np.ones((11, 1, 1), dtype=np.float32)  # Dimensions of inverse coefficients are 11x4096x4096.
        rfp_invlin = InverseLinearity(None, meta_data=invlin_meta, inv_coeffs=test_data)
        rfp_invlin.make_inverselinearity_obj()

        # Build reference asdf file object and test by asserting validate returns none.
        rfp_test_invlin = rds.InverseLinearityRef()
        rfp_test_invlin['meta'] = rfp_invlin.meta
        rfp_test_invlin['coeffs'] = rfp_invlin.inv_coeffs
        rfp_test_invlin['dq'] = rfp_invlin.mask

        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_test_invlin}
        # The validate method will return a list of exceptions the schema failed to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert tf.validate() is None

