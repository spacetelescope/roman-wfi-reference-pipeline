"""
Detector response kernels applied to oversampled STPSF models to make ePSFs.
 
Only numpy/scipy are needed here so the kernels can be unit tested without
stpsf, stsynphot or CRDS.
 
The ePSF convention after disucssions with T. Brandt and E. Schlafly each value 
is the flux a native pixel would collect if it were centred at that oversampled 
position. With the input PSF normalized to ~1, the pixel-response-convolved PSF 
therefore sums to ~oversample**2.
"""
 
import logging
 
import numpy as np
from scipy import ndimage
 
log = logging.getLogger(__name__)
 
 
def build_prf_kernel(oversample):
    """
    Pixel response (top-hat) kernel on the oversampled grid.
 
    A native pixel centred on an oversampled pixel spans ``oversample - 1``
    whole sub-pixels plus half of the sub-pixel on each side, so the 1-D
    kernel is ``[0.5, 1, ..., 1, 0.5]`` with length ``oversample + 1``.
 
    * Sums to ``oversample**2`` (full native-pixel flux).
    * Odd-sized and symmetric, so convolving does not shift the PSF centre.
    * Identical to a box of ones of size ``oversample`` followed by a
      half-pixel linear interpolation back onto the original grid.
    """
    if not isinstance(oversample, (int, np.integer)) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
 
    k1d = np.ones(oversample + 1)
    k1d[0] = k1d[-1] = 0.5
    return np.outer(k1d, k1d)
 
 
def build_ipc_kernel(ipc_kernel, oversample, normalize=True):
    """
    Spread a native-pixel IPC kernel (3x3) onto the oversampled grid.
 
    The kernel is renormalized to sum to 1 by default, so that a small
    normalization error in the IPC reference file is not carried into the
    ePSF flux.
    """
    ipc_kernel = np.asarray(ipc_kernel, dtype=np.float64)
    if ipc_kernel.ndim != 2 or any(n % 2 == 0 for n in ipc_kernel.shape):
        raise ValueError(f"IPC kernel must be 2-D and odd-sized, got {ipc_kernel.shape}")
 
    total = ipc_kernel.sum()
    if normalize:
        if not np.isclose(total, 1.0, rtol=1e-6):
            log.info(f"IPC kernel sums to {total:.6f}; renormalizing to 1.")
        ipc_kernel = ipc_kernel / total
 
    ny, nx = ipc_kernel.shape
    sparse = np.zeros(((ny - 1) * oversample + 1, (nx - 1) * oversample + 1))
    sparse[::oversample, ::oversample] = ipc_kernel
    return sparse
 
 
def apply_prf(psf_array, oversample=4):
    """Convolve an oversampled PSF with the pixel response kernel."""
    return ndimage.convolve(
        np.asarray(psf_array, dtype=np.float64),
        build_prf_kernel(oversample),
        mode="constant",
        cval=0,
    )
 
 
def apply_ipc(psf_array, oversample=4, ipc_kernel=None):
    """
    Convolve an oversampled PSF with the (normalized) IPC kernel.
 
    Returns the input unchanged (as float64) when ``ipc_kernel`` is None.
    """
    psf_array = np.asarray(psf_array, dtype=np.float64)
    if ipc_kernel is None:
        return psf_array
    return ndimage.convolve(
        psf_array,
        build_ipc_kernel(ipc_kernel, oversample),
        mode="constant",
        cval=0,
    )