import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from soc_roman_tools.siaf import siaf
try:
    import sep
    use_sep = True
except ImportError:
    from photutils.detection import DAOStarFinder
    use_sep = False
from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u

# For now we will read a text file, later we will move to the database
import astropy.table

def read_refcat(refcat_path, img_path, output_path=None, nsigma=5, fwhm=3):
    """
    Function that reads a reference catalog and an image and produces
    a matched catalog that can be used to compute the distortion coefficients

    Inputs
    ------
    refcat_path (string): Path to the reference catalog to read.
    img_path (string): Path to the image to use for calibration.
    output_path (string): Path to write the catalog (optional).
    nsigma (int): SNR threshold for detection algorithm.
    fwhm (int): FWHM value to use for photutils.detection.DAOStarFinder (in units of pixels).
        Ignored if `sep` is used.

    Returns
    -------
    refcat_out (dict): Dictionary containing the coordinates of the detected objects
        from the reference catalog, and their matches in the Idl frame.
    """

    # WILL NEED TO UPDATE THE CATALOG READ!!
    refcat = astropy.table.Table.read(refcat_path)
    # AND LIKELY THE IMAGE, TOO
    img = fits.open(img_path)[0].data
    # Read SIAF -- will need to convert from sky -> det for the matching, and from sky -> Idl to feed the fit
    sfile = siaf.RomanSiaf().read_roman_siaf()
    
    if use_sep:
        # SEP is picky with endianness
        bkg = sep.Background(img.byteswap().newbyteorder())
        objects = sep.extract(img.byteswap().newbyteorder()-bkg, nsigma, err=bkg.globalrms)
        x_sci = objects['x']
        y_sci = objects['y']
        # TODO include additional selection cuts
    else:
        mean, median, std = sigma_clipped_stats(img)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=nsigma*std)
        objects = daofind(img - median)
        x_sci = objects['xcentroid']
        y_sci = objects['ycentroid']
