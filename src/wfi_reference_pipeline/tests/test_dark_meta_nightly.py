import unittest, yaml, asdf
from wfi_reference_pipeline.dark import dark
from ..utilities.reference_file import ReferenceFile
import importlib.resources as importlib_resources
import roman_datamodels.stnode as rds
import numpy as np


# def make_dark_test_meta():
#     # Load default dark meta data
#     with importlib_resources.path('wfi_reference_pipeline.resources.data', 'meta_dark_WIM.yaml') as afile:
#         with open(afile) as af:
#             test_meta = yaml.safe_load(af)
#
#     test_ma_table_id = 0
#     test_tablename = 'Test Table'
#     test_ngroups = 1
#     test_nframes = 1
#     test_groupgap = 0
#
#     test_meta['instrument'].update({'name': f'WFI'})
#     det = 1
#     test_meta['instrument'].update({'detector': f'WFI{det:02d}'})
#     test_meta['instrument'].update({'optical_element': 'F158'})
#     dummy_date = '2020-01-01T00:00:00.000'
#     test_meta.update({'useafter': dummy_date})
#     test_meta['exposure'].update({'ngroups': test_ngroups, 'nframes': test_nframes, 'groupgap': test_groupgap,
#                                   'ma_table_name': test_tablename, 'ma_table_number': test_ma_table_id})
#     test_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
#     test_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
#     test_meta.update({'description': 'For nightly pytests.'})
#
#     return test_meta, test_ngroups, test_nframes
#
#
# def make_test_dark(test_meta, test_ngroups, test_nframes):
#
#     """
#     Test function that detects the length of the datacube to
#     use during the fits.
#     """
#
#     test_file = '/grp/roman/RFP/TEST/scratch/nightly_test_dummy_dark.asdf'
#     test_data = np.ones((3, 3, 3), dtype=np.float32)
#     test_dark = dark.Dark(None, meta_data=test_meta, outfile=test_file, clobber=True, dark_read_cube=test_data)
#     test_dark.make_ma_table_dark(test_ngroups, test_nframes)
#
#     return test_dark
#
#
# class NightlyMetaDarkTest(unittest.TestCase):
#     """
#     Test function that detects the length of the datacube to
#     use during the fits.
#     """
#     def test_rfp_dark_meta_nightly(self, test_dark):
#
#         """
#         Test function that detects the length of the datacube to
#         use during the fits.
#         """
#
#         rfp_dummy_dark = rds.DarkRef()
#         rfp_dummy_dark['data'] = test_dark.resampled_dark_cube
#         rfp_dummy_dark['err'] = test_dark.resampled_dark_cube_err
#         rfp_dummy_dark['dq'] = test_dark.mask
#         rfp_dummy_dark['meta'] = test_dark.meta
#         tf = asdf.AsdfFile()
#         tf.tree = {'roman': rfp_dummy_dark}
#         assert tf.validate() is None


# Load default dark meta data
with importlib_resources.path('wfi_reference_pipeline.resources.data', 'meta_dark_WIM.yaml') as afile:
    with open(afile) as af:
        test_meta = yaml.safe_load(af)

test_ma_table_id = 0
test_tablename = 'Test Table'
test_ngroups = 1
test_nframes = 1
test_groupgap = 0

test_meta['instrument'].update({'name': f'WFI'})
det = 1
test_meta['instrument'].update({'detector': f'WFI{det:02d}'})
test_meta['instrument'].update({'optical_element': 'F158'})
dummy_date = '2020-01-01T00:00:00.000'
test_meta.update({'useafter': dummy_date})
test_meta['exposure'].update({'ngroups': test_ngroups, 'nframes': test_nframes, 'groupgap': test_groupgap,
                              'ma_table_name': test_tablename, 'ma_table_number': test_ma_table_id})
test_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
test_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
test_meta.update({'description': 'For nightly pytests.'})


test_file = '/grp/roman/RFP/TEST/scratch/nightly_test_dummy_dark.asdf'
test_data = np.ones((3, 3, 3), dtype=np.float32)
test_dark = dark.Dark(None, meta_data=test_meta, outfile=test_file, clobber=True, dark_read_cube=test_data)
test_dark.make_ma_table_dark(num_res, wfi_mode='WIM')


rfp_dummy_dark = rds.DarkRef()
rfp_dummy_dark['data'] = test_dark.resampled_dark_cube
rfp_dummy_dark['err'] = test_dark.resampled_dark_cube_err
rfp_dummy_dark['dq'] = test_dark.mask
rfp_dummy_dark['meta'] = test_dark.meta
tf = asdf.AsdfFile()
tf.tree = {'roman': rfp_dummy_dark}
assert tf.validate() is None




