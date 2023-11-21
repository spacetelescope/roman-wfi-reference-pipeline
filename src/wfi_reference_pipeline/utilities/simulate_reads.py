import numpy as np


def simulate_reads(n_reads, exptime, darkrate, darkvar):

    """ The method SimulateReads is a function that takes in the number of reads,
    exptime, darkrate, darkvariance, number of hot pixels, the hot pixel magnitude,
    variance in hot pixels so create or simulate a series of reads.

    Parameters - External
    ----------
    n_reads = total number of reads in a cube
    exp_time = is the exposure time or frame read time
    darkrate = the number of electrons per pixel per second
    darkvar = variance in the darkrate away from its nominal value

    Parameters - Internal
    ----------
    num_hp = total number of hotpixels, recall 4088x4088 ~ 16Mpixels
    hp_mag = magnitude of hot pixels
    hp_var = variance in hotpixels away from hp_mag

    Returns
    -------
    read_cube: cube of reads to make darks
    """

    print("Simulating WFI dark reads...")
    rate_image = np.random.normal(darkrate, scale=darkvar, size=(4096, 4096))

    # place a somewhat random number of hotpixels into the rate image
    num_hotpixels = np.random.randint(1300, 1600)
    hotpixel_val = 0.015
    hotpixel_var = 0.010
    # randomly locate hot pixels on the SCA detector
    coords_x = np.random.randint(4, 4091, num_hotpixels)  # only place hot pixels on science 4088x4088 grid
    coords_y = np.random.randint(4, 4091, num_hotpixels)  # only place hot pixels on science 4088x4088 grid
    # create the randomly located hotpixels and locate within the raadcube
    hp_pix_array = np.random.normal(hotpixel_val, scale=hotpixel_var, size=(num_hotpixels))
    # update the rate image with hotpixels
    rate_image[coords_x, coords_y] = hp_pix_array

    read_cube = np.zeros((n_reads, 4096, 4096), dtype=np.float32) # initialize read cube
    for read_r in range(0, n_reads):
        # create read cube by simulating data in reads
        read_cube[read_r, :, :] = (read_r + 1) * exptime * (rate_image +
                                                            np.random.randint(0, 10, size=(4096, 4096))/50000.)
                        # this last term may be read noise but is set low intentionally to check

    # return read_cube
    return read_cube, rate_image
