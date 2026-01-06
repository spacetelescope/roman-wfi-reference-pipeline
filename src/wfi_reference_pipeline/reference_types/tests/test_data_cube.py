import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    SCI_PIXEL_X_COUNT,
    SCI_PIXEL_Y_COUNT,
    WFI_TYPE_IMAGE,
    WFI_TYPE_PRISM,
)
from wfi_reference_pipeline.reference_types.data_cube import DataCube


def test_create_valid_datacube_4k_image():
    f, x, y = 3, SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT
    wfi_type = WFI_TYPE_IMAGE

    test_data = np.zeros((f, x, y))

    datacube = DataCube(test_data, wfi_type)

    assert datacube is not None 
    assert datacube.time_array is not None
    assert len(datacube.time_array) == f
    assert datacube.num_reads == f
    assert datacube.num_i_pixels == x
    assert datacube.num_j_pixels == y

def test_create_valid_datacube_4k_prism():
    f, x, y = 3, SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT
    wfi_type = WFI_TYPE_PRISM

    test_data = np.zeros((f, x, y))

    datacube = DataCube(test_data, wfi_type)

    assert datacube is not None
    assert datacube.time_array is not None
    assert len(datacube.time_array) == f
    assert datacube.num_reads == f
    assert datacube.num_i_pixels == x
    assert datacube.num_j_pixels == y

def test_create_valid_datacube_1k_image():
    f, x, y = 1, SCI_PIXEL_X_COUNT // 4, SCI_PIXEL_Y_COUNT // 4
    wfi_type = WFI_TYPE_IMAGE

    test_data = np.zeros((f, x, y))

    datacube = DataCube(test_data, wfi_type)

    assert datacube is not None 
    assert datacube.time_array is not None
    assert len(datacube.time_array) == f
    assert datacube.num_reads == f
    assert datacube.num_i_pixels == x
    assert datacube.num_j_pixels == y

def test_invalid_datacube_lengths():
    f, x, y = 3, SCI_PIXEL_X_COUNT, SCI_PIXEL_Y_COUNT - 1
    wfi_type = WFI_TYPE_IMAGE

    test_data = np.zeros((f, x, y))

    with pytest.raises(ValueError):
        _ = DataCube(test_data, wfi_type)