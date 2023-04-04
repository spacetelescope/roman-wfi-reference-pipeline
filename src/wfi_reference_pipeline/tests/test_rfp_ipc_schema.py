import yaml, asdf, os
from wfi_reference_pipeline.utilities import ipc
import wfi_reference_pipeline.resources.data as resrouce_meta
import importlib.resources
import roman_datamodels.stnode as rds
import numpy as np
from pathlib import Path

# Load all of the yaml files with reference file specific meta data
meta_yml_fls = importlib.resources.files(resrouce_meta)


def test_rfp_ipc_schema():
    """
    Use the WFI reference file pipeline Dark() module to build a testable object which is then validated against
    the DMS dark reference file schema. The test is designed to check for dependency changes in the
    """

    # Load default dark meta from yaml file resource
    # This doesn't work for some reason. Trying another approach.

    yaml_file_path = Path("/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/src/wfi_reference_pipeline/"
                          "resources/data/meta_base.yaml")

    # Load the YAML file contents into a dictionary using safe_load()
    with yaml_file_path.open() as af:
        test_meta = yaml.safe_load(af)

        if test_meta is None:
            raise ValueError(f'No meta data loaded from yaml file.')
        else:
            # Update dark meta data.
            instrument = {'name': 'WFI', 'detector': 'WFI01', 'optical_element': 'F158'}  # iterate through detectors XX
            test_meta['instrument'] = instrument
            test_meta.update({'useafter': '2020-01-01T00:00:00.000'})
            test_meta.update({'description': 'For schema pytest validation.'})
            test_meta['pedigree'] = "DUMMY"
            test_meta['description'] = "Updated reference files for CRDS 20220615. For Build 22Q3_B6 testing."

            #
            test_data = np.ones((3, 3), dtype=np.float32)
            rfp_ipc = ipc.IPC(meta_data=test_meta, user_ipc=test_data)
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

