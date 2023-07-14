import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.inverselinearity.inverselinearity import InverseLinearity


    def test_rfp_inverselinearity_schema():
        """
        Use the WFI reference file pipeline InvLinearity() module to build a testable object which is then validated against
        the DMS reference file schema.
        """

        # Make reftype specific data class object and export meta data as dict.
        tmp = MakeTestMeta(ref_type='INVERSELINEARITY')
        inverselinearity_test_meta = tmp.meta_inverselinearity.export_asdf_meta()

        # Make RFP Inverse Linearity reference file object for testing.
        test_data = np.ones((11, 1, 1), dtype=np.float32)  # Dimensions of inverse coefficients are 11x4096x4096.
        rfp_inverselinearity = InverseLinearity(None, meta_data=inverselinearity_test_meta, inv_coeffs=test_data)

        # Build reference type object using meta data from the RFP
        invlin = rds.InverseLinearityRef()
        invlin['meta'] = rfp_inverselinearity.meta
        invlin['coeffs'] = test_data
        invlin['dq'] = rfp_inverselinearity.mask
        tf = asdf.AsdfFile()
        tf.tree = {'roman': invlin}
        # The validate method will return a list of exceptions the schema failed to validate against
        # the json schema in DMS. If none, then validate == TRUE.
        assert tf.validate() is None

