import numpy as np
import random
import logging
from wfi_reference_pipeline.constants import WFI_FRAME_TIME, WFI_MODE_WIM, WFI_MODE_WSM


def simulate_dark_reads(n_reads, ni=4096, exp_time=WFI_FRAME_TIME[WFI_MODE_WIM], dark_rate=0.005, dark_var=0.001,
                        hot_pix_rate=0.015, hot_pix_var=0.010, num_hot_pix=2000, num_hot_pix_var=200,
                        warm_pix_rate=0.050, warm_pix_var=0.010, num_warm_pix=1000, num_warm_pix_var=100,
                        dead_pix_rate=0.0001, dead_pix_var=0.00001, num_dead_pix=500, num_dead_pix_var=50,
                        noise_mean=0.001, noise_var=0.0005):
    """
    Function to create a dark read cube with random number of hot, warm, and dead pixels.

    Parameters
    ----------
    n_reads: int;
        The number of reads to be simulated into a cube of n_reads x ni x ni.
    ni: int: default = 4096;
        The number of x=y pixels to be simulated in the square array.
    exp_time: float; default = WFI_FRAME_TIME[WFI_MODE_WIM]
        WIM exposure time is set to default from constants.py in seconds.
        WIM exp_time = 3.04 seconds, WSM exp_time = 4.03 seconds
    dark_rate: float; default = 0.005
        The simulated detector dark rate in e/p/s electrons per pixel per second.
    dark_var: float; default = 0.001
        The variance in the simulated dark rate.
    hot_pix_rate: float; default = 0.015
        The simulated hot pixel rate in e/p/s.
    hot_pix_var: float; default = 0.010
        The variance in the simulated hot pixel rate.
    num_hot_pix: int; default = 2000
        Average random number of hot pixels injected into dark rate image.
    num_hot_pix_var: int; default = 200
        The variance in the random number of hot pixels.
    warm_pix_rate: float; default = 0.050
        The simulated warm pixel rate in e/p/s.
    warm_pix_var: float; default = 0.010
        The variance in the simulated warm pixel rate
    num_warm_pix: int; default = 1000
        Average random number of warm pixels injected into dark rate image.
    num_warm_pix_var: int; default = 100
        The variance in the random number of warm pixels.
    dead_pix_rate: float; default = 0.0001
        The simulated dead pixel rate in e/p/s.
    dead_pix_var: float; default = 0.00001
        The variance in the simulated dead pixel rate.
    num_dead_pix: int; default = 500
        Average random number of dead pixels injected into dark rate image.
    num_dead_pix_var: int; default = 50
        The variance in the random number of dead pixels.
    noise_mean: float; default = 0.001
        The average noise value int he dark rate image.
    noise_var: float; default = 0.0005
        The variance in the dark rate image.

    Returns
    ----------
    read_cube [n_reads, 4096, 4096], rate_image [4096, 4096]
    """

    logging.info('Making dark read cube.')
    if exp_time == WFI_FRAME_TIME[WFI_MODE_WIM]:
        print("Making WFI Imaging Mode (WIM) dark read cube with exposure time", exp_time, "seconds.")
    elif exp_time == WFI_FRAME_TIME[WFI_MODE_WSM]:
        print("Making WFI Spectral Mode (WSM) dark read cube with exposure time", exp_time, "seconds.")
    else:
        raise ValueError('Invalid WFI exposure time.')

    # Initialize rate image.
    rate_image = np.random.normal(dark_rate, scale=dark_var, size=(ni, ni))

    # Get number and locations of hot pixels and apply to rate image.
    hot_pix_count = round(random.gauss(num_hot_pix, num_hot_pix_var))
    coords_x = np.random.randint(4, ni-5, hot_pix_count)
    coords_y = np.random.randint(4, ni-5, hot_pix_count)
    hot_pixels = np.random.normal(hot_pix_rate, scale=hot_pix_var, size=hot_pix_count)
    rate_image[coords_x, coords_y] = hot_pixels

    # Get number and locations of warm pixels and apply to rate image.
    warm_pix_count = round(random.gauss(num_warm_pix, num_warm_pix_var))
    coords_x = np.random.randint(4, ni-5, warm_pix_count)
    coords_y = np.random.randint(4, ni-5, warm_pix_count)
    warm_pixels = np.random.normal(warm_pix_rate, scale=warm_pix_var, size=warm_pix_count)
    # Over writing any hot pixels as warm pixels.
    rate_image[coords_x, coords_y] = warm_pixels

    # Get number and locations of dead pixels and apply to rate image.
    dead_pix_count = round(random.gauss(num_dead_pix, num_dead_pix_var))
    coords_x = np.random.randint(4, ni-5, dead_pix_count)
    coords_y = np.random.randint(4, ni-5, dead_pix_count)
    dead_pixels = np.random.normal(dead_pix_rate, scale=dead_pix_var, size=dead_pix_count)
    # Over writing any hot pixels or warm pixels as dead pixels.
    rate_image[coords_x, coords_y] = dead_pixels

    read_cube = np.zeros((n_reads, ni, ni), dtype=np.float32)  # Initialize read cube
    for read_r in range(0, n_reads):
        # Create read cube by simulating data in reads and add noise.
        rn = np.random.normal(noise_mean, noise_var, size=(ni, ni))  # Random noise term to add; simulate a read noise.
        read_cube[read_r, :, :] = (read_r + 1) * exp_time * rate_image + rn
    return read_cube, rate_image
