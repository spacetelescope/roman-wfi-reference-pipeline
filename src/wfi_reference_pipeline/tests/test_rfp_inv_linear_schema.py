import asdf
from wfi_reference_pipeline.inv_linearity import inv_linearity
from wfi_reference_pipeline.tests import make_test_meta
import roman_datamodels.stnode as rds
import numpy as np


def test_rfp_inv_linear_schema():
    """
    Use the WFI reference file pipeline InvLinearity() module to build a testable object which is then validated against
    the DMS reference file schema.
    """
    invlin_test = make_test_meta.MakeTestMeta(ref_type='INVERSELINEARITY')

    if invlin_test.test_meta is None:
        raise ValueError(f'No meta to test.')
    else:
        # Make RFP Inverse Linearity reference file object for testing.
        test_data = np.ones((11, 1, 1), dtype=np.float32)  # Dimensions of inverse coefficients are 11x4096x4096.
        rfp_invlin = inv_linearity.InvLinearity(None, meta_data=invlin_test.test_meta, inv_coeffs=test_data)
        rfp_invlin.make_inv_linearity_obj()

        # Build reference asdf file object and test by asserting validate returns none.
        rfp_test_invlin = rds.InverseLinearityRef()
        rfp_test_invlin['meta'] =  rfp_invlin.meta
        rfp_test_invlin['coeffs'] = rfp_invlin.inv_coeffs
        rfp_test_invlin['dq'] = rfp_invlin.mask

        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_test_invlin}
        # The validate method will return a list of exceptions the schema failed to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert tf.validate() is None

