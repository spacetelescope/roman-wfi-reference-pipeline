import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.referencepixel.referencepixel import ReferencePixel


    def test_rfp_referencepixel_schema():
        """
        Use the WFI reference file pipeline ReferencePixel() module to build a testable object which is then
        validated against the DMS reference file schema.
        """

        # Make reftype specific data class object and export meta data as dict.
        tmp = MakeTestMeta(ref_type='REFPIX')
        referencepixel_test_meta = tmp.meta_referencepixel.export_asdf_meta()

        # Make RFP Reference Pixel reference file object for testing.
        test_data = np.ones((1, 1), dtype=np.complex128)  # Dimensions of inverse coefficients are 11x4096x4096.
        rfp_referencepixel = ReferencePixel(None, meta_data=referencepixel_test_meta)

        # Build reference type object using meta data from the RFP
        refpix = rds.RefpixRef()
        refpix['meta'] = rfp_referencepixel.meta
        refpix['gamma'] = test_data
        refpix['zeta'] = test_data
        refpix['alpha'] = test_data
        tf = asdf.AsdfFile()
        tf.tree = {'roman': refpix}
        # The validate method will return a list of exceptions the schema failed to validate against
        # the json schema in DMS. If none, then validate == TRUE.
        assert tf.validate() is None

