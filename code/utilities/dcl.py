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
        integration, and another called 'zeroread' that is
        the first read of the integration.
    """

    # Read in the FITS file. Squeeze it to remove
    # the superfluous fourth dimension.
    with fits.open(file) as hdu:
        integration = np.squeeze(hdu[1].data)

    # Re-order the array to increasing reads along
    # the z-axis.
    integration = integration[::-1, :, :]

    # Separate out the zero read so it can be used
    # if needed, but doesn't get included in the
    # rest of the data.
    data = {'reads': integration[1:, :, :],
            'zeroread': integration[0, :, :]}

    return data
