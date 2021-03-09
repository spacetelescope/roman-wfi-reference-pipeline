"""
Tools for working with DCL data. Mainly FITS handling.
"""

from astropy.io import fits
import numpy as np


def read_dcl_data(file):
    """
    Read a DCL FITS file and return the data in an
    organized way that is easier to work with.

    Inputs
    ------
    file (string): DCL FITS file name.

    Returns
    -------
    data (dict): A dictionary containing two numpy.ndarrays,
        one called 'reads' containing the data cube of the
        integration, another called 'zeroread' that is
        the first read of the integration, and finally
        'header' that includes the primary FITS header.
    """

    # Read in the FITS file. Squeeze it to remove
    # the superfluous fourth dimension.
    with fits.open(file) as hdu:
        integration = np.squeeze(hdu[1].data)
        hdr = hdu[0].header

    # Re-order the array to increasing reads along
    # the z-axis.
    integration = integration[::-1, :, :]

    # Separate out the zero read so it can be used
    # if needed, but doesn't get included in the
    # rest of the data.
    data = {'reads': integration[1:, :, :],
            'zero_read': integration[0, :, :],
            'header': hdr}

    return data


def subtract_zero_read(data):
    """
    Quickly subtract the zero read from all other reads in the integration.

    Inputs
    ------
    data (dict): Dictionary output from .read_dcl_data

    Returns
    -------
    data (dict): Updated dictionary that has the zero read subtracted from
        every other read in the integration.
    """

    data['reads'] -= data['zero_read']

    return data
