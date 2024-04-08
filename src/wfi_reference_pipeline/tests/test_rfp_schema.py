import os

# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import unittest
    import numpy as np
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.reference_types.dark.dark import Dark
    from wfi_reference_pipeline.reference_types.flat.flat import Flat
    from wfi_reference_pipeline.reference_types.gain.gain import Gain
    from wfi_reference_pipeline.reference_types.interpixelcapacitance.interpixelcapacitance import InterPixelCapacitance
    from wfi_reference_pipeline.reference_types.inverselinearity.inverselinearity import InverseLinearity
    from wfi_reference_pipeline.reference_types.linearity.linearity import Linearity
    from wfi_reference_pipeline.reference_types.mask.mask import Mask
    from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
    from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import ReferencePixel
    from wfi_reference_pipeline.reference_types.saturation.saturation import Saturation


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
            test_data = np.ones((3, 3, 3), dtype=np.float32)
            rfp_dark = Dark(None, meta_data=dark_test_meta, input_dark_cube=test_data)
            rfp_dark.initialize_arrays(num_resultants=1, ni=3)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_dark.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_flat_schema(self):
            """
            Use the WFI reference file pipeline Flat() module to build a testable object
            which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='FLAT')
            flat_test_meta = tmp.meta_flat.export_asdf_meta()

            # Make RFP Flat reference file object for testing.
            test_data = np.ones((3, 3), dtype=np.float32)
            rfp_flat = Flat(None, meta_data=flat_test_meta, input_flat_cube=test_data)
            rfp_flat.make_flat_rate_image()

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_flat.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_gain_schema(self):
            """
            Use the WFI reference file pipeline Gain() module to build a testable object
            which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='GAIN')
            gain_test_meta = tmp.meta_gain.export_asdf_meta()

            # Make RFP Gain reference file object for testing.
            test_data = np.ones((3, 3), dtype=np.float32)
            rfp_gain = Gain(test_data, meta_data=gain_test_meta)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_gain.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_interpixelcapacitance_schema(self):
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
            # If none, then datamodel tree is valid.
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
                                                    input_coefficients=test_data)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_inverselinearity.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_linearity_schema(self):
            """
            Use the WFI reference file pipeline Linearity() module to build a testable
            object which is the validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='LINEARITY')
            linearity_test_meta = tmp.meta_linearity.export_asdf_meta()

            # Make RFP Linearity reference file object for testing.
            test_data = np.ones((7, 1, 1),
                                dtype=np.float32)  # Dimensions of coefficients are 11x4096x4096.
            with self.assertRaises(ValueError):
                Linearity(test_data, meta_data=linearity_test_meta)
            rfp_linearity = Linearity(test_data, meta_data=linearity_test_meta,
                                      optical_element='F184')

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_linearity.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_mask_schema(self):
            """
            Use the WFI reference file pipeline Mask() module to build
            testable object which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='MASK')
            mask_test_meta = tmp.meta_mask.export_asdf_meta()

            # Make RFP Mask reference file object for testing.
            rfp_mask = Mask(None, meta_data=mask_test_meta)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_mask.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_readnoise_schema(self):
            """
            Use the WFI reference file pipeline ReadNoise() module to build
            testable object which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='READNOISE')
            readnoise_test_meta = tmp.meta_readnoise.export_asdf_meta()

            # Make RFP Read Noise reference file object for testing.
            test_data = np.ones((1, 1, 1),
                                dtype=np.float32)
            rfp_readnoise = ReadNoise(None, meta_data=readnoise_test_meta, input_data_cube=test_data)
            rfp_readnoise.initialize_arrays()

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_readnoise.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
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
            shape = (3,3)
            test_coeff = np.ones(shape, dtype=complex)
            rfp_referencepixel = ReferencePixel(None, meta_data=referencepixel_test_meta,
                                                alpha=test_coeff, zeta=test_coeff, gamma=test_coeff)

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_referencepixel.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None

        def test_rfp_saturation_schema(self):
            """
            Use the WFI reference file pipeline Saturation() module to build
            testable object which is then validated against the DMS reference file schema.
            """

            # Make reftype specific data class object and export meta data as dict.
            tmp = MakeTestMeta(ref_type='SATURATION')
            saturation_test_meta = tmp.meta_saturation.export_asdf_meta()

            # Make RFP Saturation reference file object for testing.
            rfp_saturation = Saturation(None, meta_data=saturation_test_meta)
            rfp_saturation.update_dq_mask()

            # Make test asdf tree
            tf = asdf.AsdfFile()
            tf.tree = {'roman': rfp_saturation.populate_datamodel_tree()}
            # Validate method returns list of exceptions the json schema file failed to match.
            # If none, then datamodel tree is valid.
            assert tf.validate() is None
