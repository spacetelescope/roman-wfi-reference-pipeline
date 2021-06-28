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

    # Subtract the cube from 65535 (for some reason?;
    # Stefano knows the answer to this...).
    # Convert to 32-bit signed integer
    # for working with data in subsequent steps.
    integration = 65535 - integration.astype(np.int32)

    data = {'reads': integration,
            'meta': hdr}

    return data
