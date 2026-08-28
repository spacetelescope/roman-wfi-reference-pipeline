"""
Tools for creating pixel area maps of the Roman WFI detectors.

Some of this code has been adapted from the JWST NIRCam code for
making their calibration reference files.
"""

import datetime
import getpass
import importlib
#import logging
from math import hypot, atan2, sin
import os
import pysiaf
import scipy

import asdf
from astropy import units as u, __version__ as astropy_version
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from roman_datamodels import stnode as rds, __version__ as rdm_version

#from .utils import logging_functions
#from .utils import matrix
#from .utils.read_siaf import get_distortion_coeffs
#from . import data as wfidata

# Automatic versioning
#from wfitools._version import version as __version__


class PixelArea:
    """
    Class for computing a pixel area map for the Roman WFI detectors.

    Inputs
    ------
    detector (integer):
        WFI detector ID number.

    siaf_file (string; default=None):
        Path to the science instrument aperture file (SIAF) containing
        the geometric distortion polynomial coefficients. If None, then
        the code will look up the copy of the SIAF included in the
        package data.

    verbose (boolean; default=False):
        Optional argument to enable logging messages with additional
        information.

    Examples
    --------
    To compute the pixel area map of the science pixels (4088 x 4088) of
    detector WFI07 that does not include the reference pixel border:

    >>> from wfitools import pixel_area_map
    >>> pam07 = pixel_area_map.PixelArea(7)
    >>> pam07.compute()
    >>> pam07.save_asdf()
    """

    def __init__(self, detector, siaf_file=None, verbose=False):

        # Set up verbosity
        self.verbose = verbose
        #if self.verbose:
        #    logging_functions.configure_logging(level=print)
        #else:
        #    logging_functions.configure_logging(level=logging.WARNING)

        # Some record-keeping attributes.
        #if not siaf_file:
        #    with importlib.resources.path(wfidata, 'roman_siaf.xml') as siaf_file:
        #        self.siaf_file = siaf_file
        #else:
        #    self.siaf_file = siaf_file
        self.siaf_file = pysiaf.Siaf('Roman')
        self.detector = f'WFI{detector:02d}'

        # Get the distortion coefficients from the SIAF.
        self.x_coeffs, self.y_coeffs = self.get_coeffs()

        self.pixel_area_map = None
        self.nominal_pixel_area = None

        print(f'Set up to create pixel area map for {self.detector}.')
        #print(f'wfitools version = {__version__}')
        print(f'roman_datamodels version = {rdm_version}')
        print(f'astropy version = {astropy_version}')
        print(f'numpy version = {np.__version__}')

    # @logging_functions.timeit
    def compute(self, include_border=False, refpix_area=False):
        """
        Purpose
        -------
        Perform the steps necessary to construct the pixel area map.

        Inputs
        ------
        include_border (boolean; default=False):
            Include the 4 pixel reference pixel border around the science
            pixel array. This makes the pixel area map have dimensions
            of 4096 x 4096. It is recommended to leave this set to False.

        refpix_area (boolean; default=False):
            If the reference pixel border is included, then this parameter
            indicates whether or not to set the area of the reference
            pixels to zero. A value of True will compute the area of the
            reference pixels. Default is False.

        Returns
        -------
        None
        """

        pixel_area_a2 = self.get_nominal_area()

        # Make a grid of pixel positions from from -det_size to +det_size.
        # Really this is half of the detector size, so that it's the full
        # detector, but centered at 0 (the reference pixel position).
        #
        # If for some reason the user wants to include the border of reference
        # pixels, we can do that here...might be useful if someone skipped
        # trimming them.
        if include_border:
            #logging.info('Including reference pixel border in pixel area '
            #             'map. Pixel area map will have dimensions 4096 x '
            #             '4096.')
            #logging.warning('!!')
            #logging.warning('Calibrated Roman WFI data have the reference '
            #                'pixel border trimmed, and dimensions of 4088 x '
            #                '4088. Do not use this pixel area map unless '
            #                'the calibrated data include the reference '
            #                'pixel border.')
            #logging.warning('!!')
            det_size = 2048
        else:
            det_size = 2044
        pixels = np.mgrid[-det_size:det_size, -det_size:det_size]
        y = pixels[0, :, :]
        x = pixels[1, :, :]

        self.pixel_area_map = jacob(self.x_coeffs,
                                           self.y_coeffs, x, y, order=5).astype(np.float32)

        # Sanity check.
        ratio = self.pixel_area_map[det_size, det_size] / pixel_area_a2.value
        logging.info(f'Jacobian determinant at (x, y) = (2044, 2044) is '
                     f'{self.pixel_area_map[det_size, det_size]} arcsec^2')
        logging.info(f'Nominal pixel area is {pixel_area_a2.value} '
                     f'arcsec^2')
        logging.info(f'Ratio (Jacobian / nominal) = {ratio}')

        # Normalize the pixel area map to the nominal pixel area.
        # Both are in units of arcseconds before the normalization.
        self.pixel_area_map /= pixel_area_a2.value

        # If the reference pixel border was included, check if we're
        # setting the area of the reference pixels to zero. This is the
        # default behavior...someone might override it.
        if include_border:
            if not refpix_area:
                logging.info(f'Reference pixel border was included, but '
                             f'refpix_area = {refpix_area}. Setting area '
                             f'of reference pixels to zero.')
                self.pixel_area_map[:4, :] = 0
                self.pixel_area_map[-4:, :] = 0
                self.pixel_area_map[:, :4] = 0
                self.pixel_area_map[:, -4:] = 0

        # Save the nominal pixel area in units of steradians.
        self.nominal_pixel_area = pixel_area_a2.to(u.sr)

    def save_asdf(self, filename=None, meta_data_override={}):
        """
        Purpose
        -------
        Write the pixel area map to an ASDF file using the 
        AREA reference file datamodel.
        
        Inputs
        ------
        filename (string; default=None):
            Name of the output ASDF file. If None, then
            construct a file name of the form
            'roman_{detector}_YYYYMMDD_hhmmss_area.asdf'.
            For example:
                roman_wfi16_20220211_222140_area.asdf
            is a pixel area map of detector WFI16 that
            was made on February 11, 2022 at 22:21:40.

        meta_data_override (dictionary; default=None):
            A dictionary of values to override default
            entries in the output ASDF file meta data.

        Returns
        -------
        None

        Examples
        --------
        To generate a pixel area map of WFI16 and save the output
        to an ASDF file, while overriding the meta data to set the
        origin to STScI and the useafter to 2022-01-01 00:00:00:

        >>> from datetime import datetime
        >>> from wfitools import pixel_area_map
        >>> from astropy.time import Time
        >>> pam16 = pixel_area_map.PixelArea(16)
        >>> pam16.compute()
        >>> useafter = Time(datetime(2022, 1, 1, 0, 0, 0))
        >>> pam16.save_asdf(meta_data_override={'origin': 'STScI', 'useafter': useafter})
        """

        if not filename:
            date = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
            filename = f'roman_{self.detector.lower()}_{date}_area.asdf'

        dm = rds.PixelareaRef()

        meta = {'reftype': 'AREA',
                'description': 'Roman WFI pixel area map.',
                'pedigree': 'GROUND',
                'telescope': 'ROMAN',
                'origin': 'STSCI/SOC',
                'author': f'wfitools version {__version__}',
                'useafter': Time(datetime.datetime(2020, 1, 1, 0, 0, 0)),
                'photometry':
                    {'pixelarea_arcsecsq': self.nominal_pixel_area.to(u.arcsec * u.arcsec).value,
                     'pixelarea_steradians': self.nominal_pixel_area.value},
                'instrument':
                    {'optical_element': 'F158',
                     'detector': self.detector.upper()},
                'pixel_scale': pixel_scale,
                'x_offset': x_offset,
                'y_offset': y_offset
                }

        # If the user wants to override any of these defaults, then do so now.
        meta.update(meta_data_override)

        #logging.info('Saving ASDF file.')
        print(f'ASDF meta data: {meta}')

        dm['data'] = self.pixel_area_map
        dm['meta'] = meta

        # Add additional layers here
        # dm['x_corrected'] = x_corrected_array
        # dm['y_corrected'] = y_corrected_array
        # And so on...

        asdf_file = asdf.AsdfFile()
        asdf_file.tree = {'roman': dm}
        asdf_file.write_to(filename)

        print(f'Pixel area map saved to file {filename}')

    def show_map(self, filename=None):
        """
        Purpose
        -------
        Display the pixel area map.

        Inputs
        ------
        filename (string; default=None):
            Name of the file to save the pixel area map image. If None,
            the image will be displayed on the screen instead.

        Returns
        -------
        None
        """

        fig = plt.figure('Pixel Area Map')
        ax = fig.add_subplot()
        img = ax.imshow(self.pixel_area_map, origin='lower',
                        vmin=np.min(self.pixel_area_map[self.pixel_area_map > 0]))
        ax.set_xlabel('X science coordinate (pixels)')
        ax.set_ylabel('Y science coordinate (pixels)')
        ax.set_title(f'Pixel Area Map for {self.detector}')
        plt.colorbar(img)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def get_nominal_area(self):
        """
        Purpose
        -------
        Compute the nominal pixel area at the reference position.

        Inputs
        ------
        None

        Returns
        -------
        pixel_area (`~astropy.units.Quantity`):
            The area of the nominal reference pixel in units of square
            arcseconds.
        """

        x_scale = hypot(self.x_coeffs[1], self.y_coeffs[1])
        y_scale = hypot(self.x_coeffs[2], self.y_coeffs[2])
        bx = atan2(self.x_coeffs[1], self.y_coeffs[1])

        pixel_area = x_scale * y_scale * sin(bx) * u.arcsec * u.arcsec

        return pixel_area

    def get_coeffs(self):
        """
        Purpose
        -------
        Get the geometric distortion polynomial coefficients from the SIAF.

        Inputs
        ------
        None

        Returns
        x_coeffs (`~numpy.ndarray`):
            Array of polynomial coefficients describing the geometric distortion
            in the X direction.

        y_coeffs (`~numpy.ndarray`):
            Array of polynomial coefficients describing the geometric distortion
            in the Y direction.
        """

        det_name = f'{self.detector}_FULL'
        coeffs = self.siaf_file[det_name].get_polynomial_coefficients()
        x_coeffs = coeffs['Sci2IdlX']
        y_coeffs = coeffs['Sci2IdlY']

        return x_coeffs, y_coeffs


    def dpdx(a, x, y, order=5):
        """
        Purpose
        -------
        Compute the differential of a 2D polynomial with respect to the X variable.
        Assumes that coefficients are ordered as described in the JWST and Roman
        science instrument aperture files (SIAFs).
    
        Inputs
        ------
        a (iterable of floats):
            An iterable (list or array) of float values giving the coefficients of
            a 2D polynomial as a function of X and Y.
    
        x (float or array of floats):
            The X coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        y (float or array of floats):
            The Y coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        order (integer; default = 5):
            The order of the 2D polynomial. Note that for Roman WFI, this should
            always be 5.
    
        Returns
        -------
        partial_x (float):
            The partial derivative with respect to X.
        """
    
        partial_x = 0.0
        k = 1  # index for coefficients
        for i in range(1, order + 1):
            for j in range(i + 1):
                if i - j > 0:
                    partial_x += (i - j) * a[k] * x**(i - j - 1) * y**j
                k += 1
        return partial_x


    def dpdy(a, x, y, order=5):
        """
        Purpose
        -------
        Compute the differential of a 2D polynomial with respect to the Y variable.
        Assumes that coefficients are ordered as described in the JWST and Roman
        science instrument aperture files (SIAFs).
    
        Inputs
        ------
        a (iterable of floats):
            An iterable (list or array) of float values giving the coefficients of
            a 2D polynomial as a function of X and Y.
    
        x (float or array of floats):
            The X coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        y (float or array of floats):
            The Y coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        order (integer; default = 5):
            The order of the 2D polynomial. Note that for Roman WFI, this should
            always be 5.
    
        Returns
        -------
        partial_x (float):
            The partial derivative with respect to X.
        """
    
        partial_y = 0.0
        k = 1  # index for coefficients
        for i in range(1, order + 1):
            for j in range(i + 1):
                if j > 0:
                    partial_y += j * a[k] * x**(i - j) * y**(j - 1)
                k += 1
        return partial_y


    def jacob(a, b, x, y, order=5):
        """
        Purpose
        -------
        Compute the Jacobian determinant of a 2D polynomial. In principal,
        this is used to compute the area of each pixel on the sky given
        a polynomial that describes the geometric distortion.
    
        Note that the functions called (dpdx and dpdy) assume that the
        order of the polynomial coefficients is the order used in the
        JWST and Roman science instrument aperture files (SIAFs).
    
        Inputs
        ------
        a (iterable of floats):
            An iterable (list or array) of float values giving the coefficients of
            a 2D polynomial as a function of X and Y that fit the X pixel positions,
            i.e., X - Xsci = f(X, Y).
    
        b (iterable of floats):
            An iterable (list or array) of float values giving the coefficients of
            a 2D polynomial as a function of X and Y that fit the Y pixel positions,
            i.e., Y - Ysci = f(X, Y).
    
        x (float or iterable of floats):
            The X coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        y (float or iterable of floats):
            The Y coordinate(s) (with respect to the reference pixel position)
            at which to evaluate the partial derivative.
    
        order (integer; default = 5):
            The order of the 2D polynomial. Note that for Roman WFI, this should
            always be 5.
    
        Returns
        -------
        jacobian (float or iterable of floats):
            The Jacobian determinant of the 2D polynomial evaluated at one or more
            input positions. The shape of the result will match the shape of the
            input x and y variables.
    
            This is the area on the sky of a pixel at position (x, y) given a
            geometric distortion described by a 2D polynomial with coefficients
            a and b as further described in the JWST and Roman SIAF documentation.
        """
        jacobian = dpdx(a, x, y, order=order) * dpdy(b, x, y, order=order) - \
                   dpdx(b, x, y, order=order) * dpdy(a, x, y, order=order)
        jacobian = scipy.fabs(jacobian)
        return jacobian
