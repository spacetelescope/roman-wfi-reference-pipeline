import yaml, asdf, importlib.resources
from wfi_reference_pipeline.utilities import ipc
from wfi_reference_pipeline.tests import make_test_meta
import wfi_reference_pipeline.resources.data as resource_meta
import roman_datamodels.stnode as rds
import numpy as np
from pathlib import Path

# Load all of the yaml files with reference file specific meta data
meta_yml_fls = importlib.resources.files(resource_meta)


def test_rfp_ipc_schema():
    """
    Use the WFI reference file pipeline Dark() module to build a testable object which is then validated against
    the DMS dark reference file schema. The test is designed to check for dependency changes in the
    """
    ipc_test = make_test_meta.MakeTestMeta(ref_type='IPC')

    if ipc_test.test_meta is None:
        raise ValueError(f'No meta data loaded from yaml file.')
    else:
        #
        test_data = np.ones((3, 3), dtype=np.float32)
        rfp_ipc = ipc.IPC(meta_data=ipc_test.test_meta, user_ipc=test_data)
        rfp_ipc.make_ipc_ref_file()

        # Build dark reference asdf file object and test by asserting validate returns none.
        rfp_test_ipc = rds.IpcRef()
        rfp_test_ipc['data'] = rfp_ipc.ipc_kernel
        rfp_test_ipc['meta'] = rfp_ipc.meta
        td = asdf.AsdfFile()
        td.tree = {'roman': rfp_test_ipc}
        # The validate method will return a list of exceptions that the schema fails to validate on against
        # the json schema in DMS. If none, then validate == TRUE.
        assert td.validate() is None

