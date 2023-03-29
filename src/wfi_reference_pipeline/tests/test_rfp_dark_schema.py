import numpy as np
import roman_datamodels.stnode as rds
import wfi_reference_pipeline.resources.data as resource_meta
import yaml, asdf, importlib.resources
from wfi_reference_pipeline.dark import dark
from roman_datamodels import units as ru
from pathlib import Path

# Load all of the yaml files with reference file specific meta data
meta_yml_fls = importlib.resources.files(resource_meta)


def test_rfp_dark_schema():
    """
    Use the WFI reference file pipeline Dark() module to build a testable object which is then validated against
    the DMS dark reference file schema. The test is designed to check for dependency changes in the
    """

    # Load default dark meta from yaml file resource
    # This doesn't work for some reason. Trying another approach below. Is there a way to open all of the yaml files
    # in this directory as python objects such that yaml load works?
    #with importlib.resources.as_file(meta_yml_fls.joinpath('meta_dark_WIM.yaml')) as af:
        #dark_test_meta = yaml.safe_load(af)
        #print(dark_test_meta)

    dark_yaml_path = Path("/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/src/wfi_reference_pipeline/"
                          "resources/data/meta_dark_WIM.yaml")

    # Load the YAML file contents into a dictionary using safe_load()
    with dark_yaml_path.open() as af:
        dark_test_meta = yaml.safe_load(af)

        if dark_test_meta is None:
            raise ValueError(f'No meta data loaded from yaml file.')
        else:
            # Set test ma table specs.
            test_ma_table_id = 0
            test_tablename = 'Test Table'
            test_ngroups = 1
            test_nframes = 1
            test_groupgap = 0

            # Update dark meta data.
            dark_test_meta['instrument'].update({'name': f'WFI'})
            dark_test_meta['instrument'].update({'detector': f'WFI{1:02d}'})  # WFI detector WFI01
            dark_test_meta['instrument'].update({'optical_element': 'F158'})
            dark_test_meta.update({'useafter': '2020-01-01T00:00:00.000'})
            dark_test_meta['exposure'].update({'ngroups': test_ngroups, 'nframes': test_nframes,
                                               'groupgap': test_groupgap, 'ma_table_name': test_tablename,
                                               'ma_table_number': test_ma_table_id})
            dark_test_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
            dark_test_meta.update({'description': 'For schema pytest validation.'})

            # Create dark test object with the reference file pipeline dark() module with test data and error arrays.
            test_data = np.ones((3, 3, 3), dtype=np.float32) * ru.DN
            rfp_dark = dark.Dark(None, meta_data=dark_test_meta, input_dark_cube=test_data)
            rfp_dark.make_ma_table_dark(test_ngroups, num_rds_per_res=test_nframes, wfi_mode='WIM')
            rfp_dark.resampled_dark_cube *= ru.DN
            rfp_dark.resampled_dark_cube_err *= ru.DN

            # Build dark reference asdf file object and test by asserting validate returns none.
            rfp_test_dark = rds.DarkRef()
            rfp_test_dark['data'] = rfp_dark.resampled_dark_cube
            rfp_test_dark['err'] = rfp_dark.resampled_dark_cube_err
            rfp_test_dark['dq'] = rfp_dark.mask
            rfp_test_dark['meta'] = rfp_dark.meta
            td = asdf.AsdfFile()
            td.tree = {'roman': rfp_test_dark}
            # The validate method will return a list of exceptions that the schema fails to validate on against
            # the json schema in DMS. If none, then validate == TRUE.
            assert td.validate() is None




