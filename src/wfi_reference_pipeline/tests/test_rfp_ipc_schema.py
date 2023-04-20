import asdf
from wfi_reference_pipeline.utilities import ipc
from wfi_reference_pipeline.tests import make_test_meta
import roman_datamodels.stnode as rds
import numpy as np


def test_rfp_ipc_schema():
    """
    Use the WFI reference file pipeline IPC() module to build a testable object which is then validated against
    the DMS reference file schema.
    """
    ipc_test = make_test_meta.MakeTestMeta(ref_type='IPC')

    if ipc_test.test_meta is None:
        raise ValueError(f'No meta to test.')
    else:
        # Make RFP IPC reference file object for testing.
        test_data = np.ones((3, 3), dtype=np.float32)
        rfp_ipc = ipc.IPC(meta_data=ipc_test.test_meta, user_ipc=test_data)
        rfp_ipc.make_ipc_kernel()
        rfp_ipc.make_ipc_obj()

        # Build reference asdf file object and test by asserting validate returns none.
        rfp_test_ipc = rds.IpcRef()
        rfp_test_ipc['data'] = rfp_ipc.ipc_kernel
        rfp_test_ipc['meta'] = rfp_ipc.meta
        tf = asdf.AsdfFile()
        tf.tree = {'roman': rfp_test_ipc}
        # The validate method will return a list of exceptions the schema failed to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert tf.validate() is None

