import os
# TODO Enable this test once RTB-DATABASE is up and running.
ON_GITLAB_ACTIONS = "GITLAB_CI" in os.environ
if not ON_GITLAB_ACTIONS:

    import asdf
    import numpy as np
    import roman_datamodels.stnode as rds
    from wfi_reference_pipeline.tests.make_test_meta import MakeTestMeta
    from wfi_reference_pipeline.utilities.ipc import IPC


    def test_rfp_ipc_schema():
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

