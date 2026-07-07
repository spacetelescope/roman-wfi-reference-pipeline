import pytest
from unittest.mock import MagicMock

import numpy as np

from wfi_reference_pipeline.constants import REF_TYPE_EPSF, REF_TYPE_PHOTOM
from wfi_reference_pipeline.reference_types.photom.photom import (
    Photom, gain
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.resources.wfi_meta_photom import (
    WFIMetaPhotom,
)


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
PHOTOM_META = MakeTestMeta(ref_type=REF_TYPE_PHOTOM)
BAD_TEST_META = MakeTestMeta(ref_type=REF_TYPE_EPSF)



@pytest.fixture
def photom_object(mocker):
    """Fixture for initializing a Photom object with valid data."""
    mocker.patch(f'{BASE_MODULE}.gain', return_value=GAIN_DICT) 
    return Photom(meta_data=PHOTOM_META.meta_photom)


# Tests
def test_photom_instantiation_with_valid_ref_type(photom_object):
    """
    Test that Photom object is created successfully with valid reference type.
    """
    assert photom_object.meta.reference_type == REF_TYPE_PHOTOM
 
        

def test_photom_instantiation_with_invalid_metadata(mocker):
    """
    Test that Photom raises TypeError with invalid metadata type.
    """

    mocker.patch(f'{BASE_MODULE}.gain', return_value=GAIN_DICT) 

    with pytest.raises(TypeError):
        Photom(meta_data=BAD_TEST_META.meta_epsf)


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
    assert 'median_gain' in keys
    assert 'sigma_gain' in keys


def test_photom_outfile_default(photom_object):
    """
    Test that the default outfile name is correct.
    """
    assert (
        photom_object.outfile == 'roman_photom_file.asdf'
    )
