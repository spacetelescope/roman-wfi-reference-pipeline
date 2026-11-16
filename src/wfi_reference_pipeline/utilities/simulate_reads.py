import logging
import random

import numpy as np

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    SCI_PIXEL_X_COUNT,
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_MODE_WSM,
)


def _pixel_count(mean_count, count_var, rng=np.random):
    """Number of pixels to inject; YAML values like 1e6 arrive as strings or floats."""
    mean_count = int(float(mean_count))
    if float(count_var) == 0:
        return mean_count
    return max(0, round(rng.normal(mean_count, float(count_var))))

def make_dark_rate_image(
    ni=DETECTOR_PIXEL_X_COUNT,
    dark_rate=0.01,
    dark_rate_var=0.001,
    hot_pix_rate=0.25,
    hot_pix_rate_var=0.10,
    num_hot_pix=2000,
    num_hot_pix_var=0,
    warm_pix_rate=0.050,
    warm_pix_rate_var=0.010,
    num_warm_pix=1000,
    num_warm_pix_var=0,
    dead_pix_rate=0.0001,
    dead_pix_rate_var=0.00001,
    num_dead_pix=500,
    num_dead_pix_var=0,
    seed=None,
):
    
    """
    Dark rate image [ni, ni] with randomly placed hot, warm, and dead pixels.
    The parameters are the same as in simulate_dark_reads. With a seed the image,
    including where every hot, warm, and dead pixel lands, is identical on every
    call, so one detector can keep the same pixel population across exposures.
    The seed uses its own random generator and does not touch np.random's global state.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    rate_image = rng.normal(dark_rate, scale=dark_rate_var, size=(ni, ni))
    for rate, rate_var, count, count_var in (
        (hot_pix_rate, hot_pix_rate_var, num_hot_pix, num_hot_pix_var),
        (warm_pix_rate, warm_pix_rate_var, num_warm_pix, num_warm_pix_var),
        (dead_pix_rate, dead_pix_rate_var, num_dead_pix, num_dead_pix_var),
    ):
        n_pix = _pixel_count(count, count_var, rng)
        coords_x = rng.randint(0, ni, n_pix)
        coords_y = rng.randint(0, ni, n_pix)
        rate_image[coords_x, coords_y] = rng.normal(rate, scale=rate_var, size=n_pix)
    return rate_image


def simulate_dark_reads(
    n_reads,
    read_pattern=None,
    ni=DETECTOR_PIXEL_X_COUNT,
    exp_time=WFI_FRAME_TIME[WFI_MODE_WIM],
    dark_rate=0.01,
    dark_rate_var=0.001,
    hot_pix_rate=0.25,
    hot_pix_rate_var=0.10,
    num_hot_pix=2000,
    num_hot_pix_var=0,
    warm_pix_rate=0.050,
    warm_pix_rate_var=0.010,
    num_warm_pix=1000,
    num_warm_pix_var=0,
    dead_pix_rate=0.0001,
    dead_pix_rate_var=0.00001,
    num_dead_pix=500,
    num_dead_pix_var=0,
    noise_mean=0.001,
    noise_std=0.0005,
    rate_image=None,
):
    """
    Function to create a dark read cube with random number of hot, warm, and dead pixels.
    Parameters
    ----------
    n_reads: int;
        The number of reads to be simulated into a cube of n_reads x ni x ni.
        The number of reads to simulate. Must equal read_pattern[-1][-1] when read_pattern is given.
    read_pattern: list of lists; Default=None (Diagnostic table)
        Nested list of lists with integers for averaging reads into resultants.
    ni: int: default = DETECTOR_PIXEL_X_COUNT;
        The number of x=y pixels to be simulated in the square array.
    exp_time: float; default = WFI_FRAME_TIME[WFI_MODE_WIM]
        WIM exposure time is set to default from constants.py in seconds.
    dark_rate: float; default = 0.005
    dark_rate: float; default = 0.01
        The simulated detector dark rate in e/p/s electrons per pixel per second.
    dark_rate_var: float; default = 0.001
        The variance in the simulated dark rate.
    hot_pix_rate: float; default = 0.015
    hot_pix_rate: float; default = 0.25
        The simulated hot pixel rate in e/p/s.
    hot_pix_rate_var: float; default = 0.010
    hot_pix_rate_var: float; default = 0.10
        The variance in the simulated hot pixel rate.
    num_hot_pix: int; default = 2000
        Average random number of hot pixels injected into dark rate image.
    num_hot_pix_var: int; default = 0
        The variance in the random number of hot pixels.
    warm_pix_rate: float; default = 0.050
        The simulated warm pixel rate in e/p/s.
    warm_pix_rate_var: float; default = 0.010
        The variance in the simulated warm pixel rate
    num_warm_pix: int; default = 1000
        Average random number of warm pixels injected into dark rate image.
    num_warm_pix_var: int; default = 0
        The variance in the random number of warm pixels.
    dead_pix_rate: float; default = 0.0001
        The simulated dead pixel rate in e/p/s.
    dead_pix_rate_var: float; default = 0.00001
        The variance in the simulated dead pixel rate.
    num_dead_pix: int; default = 500
        Average random number of dead pixels injected into dark rate image.
    num_dead_pix_var: int; default = 0
        The variance in the random number of dead pixels.
    noise_mean: float; default = 0.001
        The average noise value in the dark rate image.
    noise_std: float; default = 0.0005
        The standard deviation in the Gaussian spread or width of
        frame noise added to the noise mean for the array.
    rate_image: ndarray [ni, ni]; default = None
        Reuse an existing dark rate image (e.g. from make_dark_rate_image with a
        detector seed) so hot, warm, and dead pixels stay in place across exposures.
        If given, the dark/hot/warm/dead pixel parameters are ignored. If None, a new
        rate image is made from those parameters.

    Returns
    ----------
    resampled_data [n_resultants, ni, ni] (or [n_reads, ni, ni] with no read_pattern), rate_image [ni, ni]
    """

    logging.info("Making dark read cube.")
    if exp_time == WFI_FRAME_TIME[WFI_MODE_WIM]:
        print(
            "Making WFI Imaging Mode (WIM) dark read cube with frame time",
            exp_time,
            "seconds.",
        )
    elif exp_time == WFI_FRAME_TIME[WFI_MODE_WSM]:
        print(
            "Making WFI Spectral Mode (WSM) dark read cube with frame time",
            exp_time,
            "seconds.",
        )
    elif exp_time == 1.0:
        print(
            "Simulating reads for diagnostic purposes with frame time",
            exp_time,
            "second.",
        )
    else:
        raise ValueError("Invalid WFI frame time.")

    if rate_image is None:
        rate_image = make_dark_rate_image(
            ni=ni,
            dark_rate=dark_rate, dark_rate_var=dark_rate_var,
            hot_pix_rate=hot_pix_rate, hot_pix_rate_var=hot_pix_rate_var,
            num_hot_pix=num_hot_pix, num_hot_pix_var=num_hot_pix_var,
            warm_pix_rate=warm_pix_rate, warm_pix_rate_var=warm_pix_rate_var,
            num_warm_pix=num_warm_pix, num_warm_pix_var=num_warm_pix_var,
            dead_pix_rate=dead_pix_rate, dead_pix_rate_var=dead_pix_rate_var,
            num_dead_pix=num_dead_pix, num_dead_pix_var=num_dead_pix_var,
        )
    elif rate_image.shape != (ni, ni):
        raise ValueError(f"rate_image shape {rate_image.shape} does not match ({ni}, {ni}).")

    def _simulate_read(read_num):
        # Signal accumulated up to this read plus a random noise term for the read.
        rn = np.random.normal(loc=noise_mean, scale=noise_std, size=(ni, ni))
        return read_num * exp_time * rate_image + rn

    if read_pattern:
        # Non-diagnostic tables: average reads into resultants.
        n_resultants = len(read_pattern)
        if n_reads != read_pattern[-1][-1]:
            # Check to make sure n_reads match the last frame in the read_pattern
            raise ValueError('n_reads does not match with the last frame number in read_pattern.')
        
        print(f"Simulating {n_reads} reads of {ni} x {ni} pixels.")
        print(f"Averaging {n_reads} reads into {n_resultants} resultants using the read pattern.")
        # Build each resultant as its reads are made, so the full
        # n_reads x ni x ni read cube is never held in memory.
        resampled_data = np.zeros((n_resultants, ni, ni), dtype=np.float32)
        for resultant_i, read_pattern_frames in enumerate(read_pattern):
            for read_num in read_pattern_frames:
                resampled_data[resultant_i] += _simulate_read(read_num)
            resampled_data[resultant_i] /= len(read_pattern_frames)
    else:
        # Diagnostic tables: every read is its own resultant.
        print(f"Simulating {n_reads} reads of {ni} x {ni} pixels.")
        resampled_data = np.zeros((n_reads, ni, ni), dtype=np.float32)
        for read_r in range(n_reads):
            resampled_data[read_r] = _simulate_read(read_r + 1) 

    return resampled_data, rate_image


def simulate_flat_reads(
    n_reads,
    ni=SCI_PIXEL_X_COUNT,
    exp_time=WFI_FRAME_TIME[
        WFI_MODE_WIM
    ],  # Assuming default exposure time as 3.04 seconds
    flat_rate=200,
    flat_rate_var=1,
    num_low_qe_pix=1000,
    num_low_qe_pix_var=0,
    low_qe_rate=0.8,
    low_qe_rate_var=0.05,
    noise_mean=0.001,
    noise_var=0.0005,
):
    """
    Function to create a flat read cube with a random number of low QE (quantum efficiency) pixels.

    Parameters
    ----------
    n_reads: int
        The number of reads to be simulated into a cube of n_reads x ni x ni.
    ni: int: default = SCI_PIXEL_X_COUNT
        The number of x=y pixels to be simulated in the square array.
    exp_time: float; default = WFI_FRAME_TIME[WFI_MODE_WIM]
        WIM exposure time is set to default from constants.py in seconds.
        WIM exp_time = 3.04 seconds, NO WSM flats
    flat_rate: float; default = 200
        The simulated detector flat field rate in e/p/s (electrons per pixel per second).
    flat_rate_var: float; default = 1
        The variance in the simulated flat rate.
    num_low_qe_pix: int; default = 1000
        Average random number of low QE pixels injected into the flat rate image.
    num_low_qe_pix_var: int; default = 0
        The variance in the random number of low QE pixels.
    low_qe_rate: float; default = 0.8
        The simulated low QE pixel rate as a fraction of the flat rate.
    low_qe_rate_var: float; default = 0.05
        The variance in the simulated low QE pixel rate.
    noise_mean: float; default = 0.001
        The average noise value in the flat rate image.
    noise_var: float; default = 0.0005
        The variance in the noise.

    Returns
    ----------
    read_cube: numpy array
        3D numpy array with shape (n_reads, ni, ni) representing the simulated flat read cube.
    rate_image: numpy array
        2D numpy array with shape (ni, ni) representing the rate image with low QE pixels.
    """

    logging.info("Making flat read cube.")
    if exp_time == WFI_FRAME_TIME[WFI_MODE_WIM]:
        print(
            "Making WFI Imaging Mode (WIM) flat read cube with exposure time",
            exp_time,
            "seconds.",
        )
    else:
        raise ValueError("Invalid WFI exposure time for flats.")

    logging.info("Making flat read cube.")
    print("Making flat read cube with exposure time", exp_time, "seconds.")

    # Initialize rate image.
    rate_image = np.random.normal(flat_rate, scale=flat_rate_var, size=(ni, ni))

    # Determine the number of low QE pixels.
    if num_low_qe_pix_var == 0:
        low_qe_pix_count = num_low_qe_pix
    else:
        low_qe_pix_count = round(random.gauss(num_low_qe_pix, num_low_qe_pix_var))

    # Get locations and apply low QE pixels to rate image.
    coords_x = np.random.randint(0, ni, low_qe_pix_count)
    coords_y = np.random.randint(0, ni, low_qe_pix_count)
    low_qe_pixels = np.random.normal(
        low_qe_rate * flat_rate,
        scale=low_qe_rate_var * flat_rate,
        size=low_qe_pix_count,
    )
    rate_image[coords_x, coords_y] = low_qe_pixels

    # Create the read cube using the rate image and noise per read.
    read_cube = np.zeros((n_reads, ni, ni), dtype=np.float32)  # Initialize read cube
    for read_r in range(n_reads):
        # Create read cube by simulating data in reads and add noise.
        rn = np.random.normal(
            noise_mean, noise_var, size=(ni, ni)
        )  # Random noise term to add; simulate a read noise.
        read_cube[read_r, :, :] = (read_r + 1) * exp_time * rate_image + rn

    return read_cube, rate_image
