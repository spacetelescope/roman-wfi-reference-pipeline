import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf, unittest
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.dark.dark import Dark
    from wfi_reference_pipeline.utilities.ipc import IPC
    from wfi_reference_pipeline.referencepixel.referencepixel import ReferencePixel
    from wfi_reference_pipeline.inverselinearity.inverselinearity import InverseLinearity
    from astropy import units as u

    class TestSchema(unittest.TestCase):
        """
        Class test suite for all RFP schema tests
        """
        def test_dark_schema(self):
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
            test_dark = rds.DarkRef()
            # TODO - do we want reference file to output data/err/dq in correct format for each of these?
            test_dark['data'] = test_data * u.DN
            test_dark['err'] = test_data * u.DN
            test_dark['dq'] = rfp_dark.mask
            test_dark['meta'] = rfp_dark.meta  # Use meta from RFP object to validate test file against schema
            td = asdf.AsdfFile()
            td.tree = {'roman': test_dark}
            # The validate method will return a list of exceptions that the schema fails to validate on against
            # the json schema in DMS. If none, then validate == TRUE.
            assert td.validate() is None

        def test_rfp_ipc_schema(self):
            """
            Use the WFI reference file pipeline IPC() module to build a testable object which is then validated against
            the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='IPC')
            ipc_test_meta = tmp.meta_ipc.export_asdf_meta()

            # Make RFP IPC reference file object for testing.
            test_data = np.ones((3, 3), dtype=np.float32)
            rfp_ipc = IPC(ipc_test_meta, user_ipc=test_data)

            # Build reference asdf file object and test by asserting validate returns none.
            ic = rds.IpcRef()
            ic['meta'] = rfp_ipc.meta
            ic['data'] = test_data
            tf = asdf.AsdfFile()
            tf.tree = {'roman': ic}
            # The validate method will return a list of exceptions the schema failed to validate on against
            # the json schema in DMS. If none, then validate == TRUE.
            assert tf.validate() is None

        def test_rfp_referencepixel_schema(self):
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

        def test_rfp_inverselinearity_schema(self):
            """
            Use the WFI reference file pipeline InverseLinearity() module to build a testable object which is then
            validated against the DMS reference file schema.
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








