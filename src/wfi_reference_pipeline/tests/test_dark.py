import unittest, yaml, asdf
from wfi_reference_pipeline.dark import dark
import importlib.resources as importlib_resources
import roman_datamodels.stnode as rds
import numpy as np

# Load default dark meta data
with importlib_resources.path('wfi_reference_pipeline.resources.data', 'meta_dark_WIM.yaml') as afile:
    with open(afile) as af:
        dummy_meta = yaml.safe_load(af)

dummy_ma_table_id = 0
dummy_tablename = 'Dumb Table'
dummy_ngroups = 1
dummy_nframes = 1
dummy_groupgap = 0

dummy_meta['instrument'].update({'name': f'WFI'})
det = 1
dummy_meta['instrument'].update({'detector': f'WFI{det:02d}'})
dummy_meta['instrument'].update({'optical_element': 'F158'})
dummy_date = '2020-01-01T00:00:00.000'
dummy_meta.update({'useafter': dummy_date})
dummy_meta['exposure'].update({'ngroups': dummy_ngroups, 'nframes': dummy_nframes, 'groupgap': dummy_groupgap,
                              'ma_table_name': dummy_tablename, 'ma_table_number': dummy_ma_table_id})
dummy_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
dummy_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
dummy_meta.update({'description': 'For nightly pytests.'})
#dummy_meta.update({'description': None})

dummy_file = '/grp/roman/RFP/TEST/scratch/nightly_test_dummy_dark.asdf'
dummy_data = np.ones((3, 3, 3), dtype=np.float32)
dummy_time_arr = np.ones(3, dtype=np.float32)
dummy_dark = dark.Dark(None, meta_data=dummy_meta, outfile=dummy_file, clobber=True, dark_read_cube=dummy_data)
dummy_dark.exp_time_arr = np.ones((3, 3), dtype=np.float32)
dummy_dark.make_ma_table_dark(3, 1)
#dummy_dark.save_dark(write_file=False)
rfp_dummy_dark = rds.DarkRef()
rfp_dummy_dark['data'] = dummy_dark.resampled_dark_cube
rfp_dummy_dark['err'] = dummy_dark.resampled_dark_cube_err
rfp_dummy_dark['dq'] = dummy_dark.mask
rfp_dummy_dark['meta'] = dummy_dark.meta
tf = asdf.AsdfFile()
tf.tree = {'roman': rfp_dummy_dark}


def test_rfp_dark():
    assert tf.validate() is None

# contact Paul Huwe on this
# # Dark Current tests
# def test_make_dark():
#     dark = utils.mk_dark(shape=(3, 20, 20))
#     assert dark.meta.reftype == 'DARK'
#     assert dark.data.dtype == np.float32
#     assert dark.dq.dtype == np.uint32
#     assert dark.dq.shape == (20, 20)
#     assert dark.err.dtype == np.float32
#     assert dark.data.unit == ru.DN
#
#     # Test validation
#     dark_model = datamodels.DarkRefModel(dark)
#     assert dark_model.validate() is None

# def setup_dummy_meta():
#     _meta = dict()
#     _meta['useafter'] = Time.now().iso
#     _meta['pedigree'] = 'DUMMY'
#     _meta['instrument'] = dict()
#     _meta['instrument']['name'] = 'WFI'
#     _meta['instrument']['detector'] = 'WFI01'
#     return _meta

# dark_file = rds.DarkRef()
# dark_file['data'] = self.resampled_dark_cube
# dark_file['err'] = self.resampled_dark_cube_err
# dark_file['dq'] = self.mask
# dark_file['meta'] = self.meta
# # Add in the meta data and history to the ASDF tree.
# af = asdf.AsdfFile()
# af.tree = {'roman': dark_file}

