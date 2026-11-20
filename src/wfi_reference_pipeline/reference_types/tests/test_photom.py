import numpy as np
import pytest

from wfi_reference_pipeline.constants import REF_TYPE_EPSF, REF_TYPE_PHOTOM
from wfi_reference_pipeline.reference_types.photom.photom import Photom
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta

BASE_MODULE = 'wfi_reference_pipeline.reference_types.photom.photom'
GAIN_DICT = {
    'WFI01': {'median': 1.5, 'std': np.float32(0.05)}, 
    'WFI02': {'median': 1.5, 'std': np.float32(0.05)}, 
    'WFI03': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI04': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI05': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI06': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI07': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI08': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI09': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI10': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI11': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI12': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI13': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI14': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI15': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI16': {'median': 1.5, 'std': np.float32(0.05)},
    'WFI17': {'median': 1.5, 'std': np.float32(0.05)},         
    'WFI18': {'median': 1.5, 'std': np.float32(0.05)}
}
PAM_DICT = {
    'WFI01': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13}, 
    'WFI02': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13}, 
    'WFI03': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI04': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI05': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI06': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI07': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI08': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI09': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI10': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI11': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI12': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI13': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI14': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI15': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI16': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},
    'WFI17': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13},         
    'WFI18': {'pixelarea_arcsecsq': 0.011, 'pixelarea_steradians': 2.789e-13}
}

PHOTOM_META = MakeTestMeta(ref_type=REF_TYPE_PHOTOM)
BAD_TEST_META = MakeTestMeta(ref_type=REF_TYPE_EPSF)



@pytest.fixture
def modulation_file(tmp_path):
    """Create a minimal empirical-correction table for PHOTOM tests."""
    modulation_file = tmp_path / 'modulation.txt'
    modulation_file.write_text(
        'filter detector Nstars Weighted_mean Weighted_mean_err '
        'Biweight_mean Biweight_mean_err Biweight_scatter Mean Median Std\n'
        'F062 1 1 1 0 1 0 0 1 1 0\n'
    )
    return str(modulation_file)


@pytest.fixture
def photom_object(mocker, modulation_file):
    """Fixture for initializing a Photom object with valid data."""
    mocker.patch(f'{BASE_MODULE}.build_gain_and_pam_dict_from_crds', return_value=[GAIN_DICT, PAM_DICT])
    return Photom(meta_data=PHOTOM_META.meta_photom, ref_type_data=modulation_file)


# Tests
def test_photom_instantiation_with_valid_ref_type(photom_object):
    """
    Test that Photom object is created successfully with valid reference type.
    """
    assert photom_object.meta.reference_type == REF_TYPE_PHOTOM
 
        

def test_photom_instantiation_with_invalid_metadata(mocker, modulation_file):
    """
    Test that Photom raises TypeError with invalid metadata type.
    """ 
    mocker.patch(f'{BASE_MODULE}.build_gain_and_pam_dict_from_crds', return_value=[GAIN_DICT, PAM_DICT])

    with pytest.raises(TypeError):
        Photom(meta_data=BAD_TEST_META.meta_epsf, ref_type_data=modulation_file)


def test_populate_datamodel_tree(photom_object):
    """
    Test that the datamodel tree is correctly populated.
    """
    data_model_tree = (
        photom_object.populate_datamodel_tree()
    )

    assert 'meta' in data_model_tree
    assert 'phot_table' in data_model_tree

    assert isinstance(data_model_tree['phot_table'], dict)

    keys = data_model_tree['meta']['instrument'].keys()
    assert 'detector' in keys
    assert 'name' in keys



def test_photom_outfile_default(photom_object):
    """
    Test that the default outfile name is correct.
    """
    assert (
        photom_object.outfile == 'roman_photom_file.asdf'
    )