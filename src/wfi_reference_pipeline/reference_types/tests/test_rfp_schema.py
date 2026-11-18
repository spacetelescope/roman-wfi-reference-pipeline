import os

import asdf
import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
)
from wfi_reference_pipeline.reference_types.boresight_alignment_matrix.boresight_alignment_matrix import (
    BoresightAlignmentMatrix,
)
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.reference_types.dark_decay_signal.dark_decay_signal import (
    DarkDecaySignal,
)
from wfi_reference_pipeline.reference_types.detector_status.detector_status import (
    DetectorStatus,
)
from wfi_reference_pipeline.reference_types.exposure_time_calculator.exposure_time_calculator import (
    ExposureTimeCalculator,
)
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.reference_types.gain.gain import Gain
from wfi_reference_pipeline.reference_types.inter_pixel_capacitance.inter_pixel_capacitance import (
    InterPixelCapacitance,
)
from wfi_reference_pipeline.reference_types.inverse_linearity.inverse_linearity import (
    InverseLinearity,
)
from wfi_reference_pipeline.reference_types.linearity.linearity import Linearity
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.reference_types.referencepixel.referencepixel import (
    ReferencePixel,
)
from wfi_reference_pipeline.reference_types.saturation.saturation import Saturation
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, too big"
)

class TestSchema():
    """
    Class test suite for all RFP schema tests
    """

    def test_rfp_bam_schema(self):
        """
        Use the WFI reference file pipeline BoresightAlignmentMatrix() module
        to build a testable object which is then validated against
        the DMS reference file schema.
        """
 
        # Make test meta
        tmp = MakeTestMeta(ref_type='BAM')
 
        # Make RFP BoresightAlignmentMatrix reference file object for testing
        rfp_bam = BoresightAlignmentMatrix(
            meta_data=tmp.meta_bam
        )
 
        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {
            'roman': rfp_bam.populate_datamodel_tree()
        }
 
        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_dark_schema(self):
        """
        Use the WFI reference file pipeline Dark() module to build a testable object
        which is then validated against the DMS reference file schema.
        """

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='DARK')
        test_data = np.ones((3, 3, 3),
                            dtype=np.float32)

        # Make RFP Dark reference file object for testing.
        rfp_dark = Dark(meta_data=tmp.meta_dark,
                        file_list=None,
                        ref_type_data=test_data)
        rfp_dark.make_rate_image_from_data_cube()

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_dark.populate_datamodel_tree()}
        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_dark_decay_schema(self):
        """
        Use the WFI reference file pipeline DarkDecaySignal() module
        to build a testable object which is then validated against
        the DMS reference file schema.
        """

        # Make test meta
        tmp = MakeTestMeta(ref_type='DARKDECAYSIGNAL')

        # Make RFP DarkDecaySignal reference file object for testing
        rfp_dark_decay = DarkDecaySignal(
            meta_data=tmp.meta_dark_decay_signal
        )

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {
            'roman': rfp_dark_decay.populate_datamodel_tree()
        }

        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_detector_status_schema(self):
        """
        Use the WFI reference file pipeline DetectorStatus() module
        to build a testable object which is then validated against
        the DMS reference file schema.
        """

        # Make test meta
        tmp = MakeTestMeta(ref_type='DETECTORSTATUS')

        # Make RFP DetectorStatus reference file object for testing
        rfp_detector_status = DetectorStatus(
            meta_data=tmp.meta_detector_status
        )

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {
            'roman': rfp_detector_status.populate_datamodel_tree()
        }

        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_exposure_time_calculator_schema(self):
        """
        Use the WFI reference file pipeline ExposureTimeCalculator()
        module to build a testable object which is then validated
        against the DMS reference file schema.
        """

        # Make test meta
        tmp = MakeTestMeta(ref_type='ETC')

        # Make RFP ExposureTimeCalculator reference file object for testing
        rfp_etc = ExposureTimeCalculator(
            meta_data=tmp.meta_etc
        )

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {
            'roman': rfp_etc.populate_datamodel_tree()
        }

        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_flat_schema(self):
        """
        Use the WFI reference file pipeline Flat() module to build a testable object
        which is then validated against the DMS reference file schema.
        """

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='FLAT')
        test_data = np.ones((3, 3), dtype=np.float32)

        # Make RFP Flat reference file object for testing.
        rfp_flat = Flat(meta_data=tmp.meta_flat,
                        file_list=None,
                        ref_type_data=test_data)
        rfp_flat.calculate_error(fill_random=True)

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

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='GAIN')
        test_data = 2*np.ones((3, 3), dtype=np.float32)

        # Make RFP Gain reference file object for testing.
        rfp_gain = Gain(meta_data=tmp.meta_gain,
                        file_list=None,
                        ref_type_data=test_data)

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

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='IPC')
        test_data = np.ones((3, 3), dtype=np.float32)

        # Make RFP IPC reference file object for testing.
        rfp_ipc = InterPixelCapacitance(meta_data=tmp.meta_ipc,
                                        ref_type_data=test_data)

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_ipc.populate_datamodel_tree()}
        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_inverse_linearity_schema(self):
        """
        Use the WFI reference file pipeline Inverse Linearity() module to build a testable
        object which is the validated against the DMS reference file schema.
        """

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='INVERSELINEARITY')

        # Make RFP Inverse Linearity reference file object for testing.
        test_data = np.ones((7, 1, 1),
                            dtype=np.float32)  # Dimensions of coefficients are 7x1x1.

        rfp_inv_lin = InverseLinearity(meta_data=tmp.meta_inverselinearity,
                                  ref_type_data=test_data
                                  )

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_inv_lin.populate_datamodel_tree()}
        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None

    def test_rfp_linearity_schema(self):
        """
        Use the WFI reference file pipeline Linearity() module to build a testable
        object which is the validated against the DMS reference file schema.
        """

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='LINEARITY')

        # Make RFP Linearity reference file object for testing.
        test_data = np.ones((7, 1, 1),
                            dtype=np.float32)  # Dimensions of coefficients are 7x1x1.

        rfp_linearity = Linearity(meta_data=tmp.meta_linearity,
                                  ref_type_data=test_data
                                  )

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

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='MASK')
        test_mask = np.zeros((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT), dtype=np.uint32)

        # Make RFP Mask reference file object for testing.
        rfp_mask = Mask(meta_data=tmp.meta_mask,
                        dark_filelist=None,
                        flat_filelist=None,
                        input_user_mask=test_mask)

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

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='READNOISE')
        test_data = np.ones((1, 1), dtype=np.float32)

        # Make RFP Read Noise reference file object for testing.
        rfp_readnoise = ReadNoise(meta_data=tmp.meta_readnoise,
                                  ref_type_data=test_data)

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

        # Make RFP Reference Pixel reference file object for testing.
        shape = (2, 4096, 4224)
        test_data = np.ones(shape, dtype=np.float32)
        rfp_referencepixel = ReferencePixel(meta_data=tmp.meta_referencepixel,
                                            ref_type_data=test_data)
        rfp_referencepixel.make_referencepixel_image(detector_name='WFI01')

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

        # Make test meta and data.
        tmp = MakeTestMeta(ref_type='SATURATION')
        test_data = np.ones((3, 3), dtype=np.float32)

        # Make RFP Saturation reference file object for testing.
        rfp_saturation = Saturation(meta_data=tmp.meta_saturation,
                                    ref_type_data=test_data)
        rfp_saturation.make_saturation_image()

        # Make test asdf tree
        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_saturation.populate_datamodel_tree()}
        # Validate method returns list of exceptions the json schema file failed to match.
        # If none, then datamodel tree is valid.
        assert tf.validate() is None
