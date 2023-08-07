import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import unittest
    import numpy as np
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.dark.dark import Dark
    from wfi_reference_pipeline.interpixelcapacitance.interpixelcapacitance import InterPixelCapacitance
    from wfi_reference_pipeline.inverselinearity.inverselinearity import InverseLinearity
    from wfi_reference_pipeline.referencepixel.referencepixel import ReferencePixel
    from wfi_reference_pipeline.linearity.linearity import Linearity

    class TestSchema(unittest.TestCase):
        """
        Class test suite for all RFP schema tests
        """

        def test_rfp_dark_schema(self):
            """
            Use the WFI reference file pipeline Dark() module to build a testable object
            which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='DARK')
            dark_test_meta = tmp.meta_dark.export_asdf_meta()

            # Make RFP Dark reference file object for testing.
            rfp_dark = Dark('test_file.txt', meta_data=dark_test_meta)
            rfp_dark.initialize_cube_arrays(num_resultants=1, ni=3)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_dark.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then validate == TRUE.
            assert tf.validate() is None

        def test_rfp_ipc_schema(self):
            """
            Use the WFI reference file pipeline IPC() module to build a testable object
            which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='IPC')
            ipc_test_meta = tmp.meta_ipc.export_asdf_meta()

            # Make RFP IPC reference file object for testing.
            test_data = np.ones((3, 3), dtype=np.float32)
            rfp_ipc = InterPixelCapacitance(ipc_test_meta, user_ipc=test_data)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_ipc.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then validate == TRUE.
            assert tf.validate() is None

        def test_rfp_linearity_schema(self):
            """
            Use the WFI reference file pipeline Linearity() module to build a testable
            object which is the validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='LINEARITY')
            linearity_test_meta = tmp.meta_linearity.export_asdf_meta()

            # Make RFP Inverse Linearity reference file object for testing.
            test_data = np.ones((11, 1, 1),
                                dtype=np.float32)  # Dimensions of coefficients are 11x4096x4096.
            rfp_inverselinearity = Linearity(None, out_meta_data=linearity_test_meta,
                                                    inv_coeffs=test_data)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_inverselinearity.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then validate == TRUE.
            assert tf.validate() is None        
        
        def test_rfp_inverselinearity_schema(self):
            """
            Use the WFI reference file pipeline InverseLinearity() module to build
            a testable object which is the validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='INVERSELINEARITY')
            inverselinearity_test_meta = tmp.meta_inverselinearity.export_asdf_meta()

            # Make RFP Inverse Linearity reference file object for testing.
            test_data = np.ones((11, 1, 1),
                                dtype=np.float32)  # Dimensions of inverse coefficients are 11x4096x4096.
            rfp_inverselinearity = InverseLinearity(None, meta_data=inverselinearity_test_meta,
                                                    inv_coeffs=test_data)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_inverselinearity.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then validate == TRUE.
            assert tf.validate() is None

        def test_rfp_referencepixel_schema(self):
            """
            Use the WFI reference file pipeline ReferencePixel() module to build
            testable object which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='REFPIX')
            referencepixel_test_meta = tmp.meta_referencepixel.export_asdf_meta()

            # Make RFP Reference Pixel reference file object for testing.
            rfp_referencepixel = ReferencePixel(None, meta_data=referencepixel_test_meta)
            rfp_referencepixel.make_referencepixel_coeffs()

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_referencepixel.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then validate == TRUE.
            assert tf.validate() is None




