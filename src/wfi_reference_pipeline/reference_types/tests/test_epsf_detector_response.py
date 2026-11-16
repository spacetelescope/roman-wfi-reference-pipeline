import numpy as np
import pytest
from scipy import signal
 
from wfi_reference_pipeline.reference_types.empirical_psf.detector_response import (
    apply_ipc,
    apply_prf,
    build_ipc_kernel,
    build_prf_kernel,
)
 
N = 101
 
 
def _gaussian_psf(x0=50.3, y0=49.6, sx=5.0, sy=3.5):
    y, x = np.indices((N, N))
    psf = np.exp(-((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2)))
    return psf / psf.sum()
 
 
def _centroid(img):
    y, x = np.indices(img.shape)
    return np.array([(img * x).sum(), (img * y).sum()]) / img.sum()
 
 
@pytest.mark.parametrize("oversample", [2, 3, 4, 5])
def test_prf_kernel_odd_symmetric_and_sums_to_oversample_squared(oversample):
    k = build_prf_kernel(oversample)
    assert k.shape == (oversample + 1, oversample + 1)
    assert np.allclose(k, k[::-1, ::-1])
    assert np.isclose(k.sum(), oversample**2)
 
 
@pytest.mark.parametrize("oversample", [2, 4])
def test_prf_kernel_equals_box_plus_half_pixel_interpolation(oversample):
    box = np.ones((oversample, oversample))
    half_shift = np.full((2, 2), 0.25)
    assert np.allclose(signal.convolve2d(box, half_shift), build_prf_kernel(oversample))
 
 
def test_apply_prf_preserves_centroid_and_scales_flux():
    psf = _gaussian_psf()
    out = apply_prf(psf, oversample=4)
    assert np.isclose(out.sum(), 16 * psf.sum(), rtol=1e-6)
    assert np.allclose(_centroid(out), _centroid(psf), atol=1e-6)
 
 
def test_ipc_kernel_is_normalized_and_sparse():
    raw = np.array([[0.001, 0.012, 0.001], [0.012, 0.95, 0.012], [0.001, 0.012, 0.001]])
    assert not np.isclose(raw.sum(), 1.0)
    k = build_ipc_kernel(raw, oversample=4)
    assert k.shape == (9, 9)
    assert np.isclose(k.sum(), 1.0)
    assert np.count_nonzero(k) == 9
    assert np.isclose(k[4, 4], raw[1, 1] / raw.sum())
 
 
def test_apply_ipc_conserves_flux_and_symmetric_kernel_keeps_centroid():
    psf = _gaussian_psf()
    raw = np.array([[0.001, 0.012, 0.001], [0.012, 0.95, 0.012], [0.001, 0.012, 0.001]])
    out = apply_ipc(psf, oversample=4, ipc_kernel=raw)
    assert np.isclose(out.sum(), psf.sum(), rtol=1e-6)
    assert np.allclose(_centroid(out), _centroid(psf), atol=1e-6)
 
 
def test_apply_ipc_none_is_identity():
    psf = _gaussian_psf()
    assert np.array_equal(apply_ipc(psf, oversample=4, ipc_kernel=None), psf)
 
 
def test_ipc_kernel_rejects_even_shape():
    with pytest.raises(ValueError):
        build_ipc_kernel(np.ones((2, 2)), oversample=4)

        