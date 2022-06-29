import numpy as np
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


class ReferenceCatalog:
    """
    Base class ReferenceCatalog() reads a reference catalog from a file or
    database and has the ability to match it to a ReferenceFile and write the
    resulting matched catalog.
    """
    def __init__(self, path):
        """
        Inputs
        ------
        path (string):
            Path to reference catalog to be read.

        """
        # This will need to be updated
        self.refcat = astropy.table.Table.read(path)

        # Initialized empty
        self.matched_cat = None

    def match_refcat(self, detector, img, nsigma_det=5, fwhm_det=3):
        """
        This method runs a detection algorithm on the data contained
        in the input ReferenceFile and matches the detected sources to
        the sources contained in this catalog. After that, it populates
        the matched_cat attribute.

        Inputs
        ------
        detector (string):
            Name of the detector for which the distortion model is constructed.
            For example: WFI01.
        img (np.array):
            image to process and match to the reference catalog.
        nsigma_det (double):
            SNR threshold for detection algorithm.
        fwhm_det (double):
            FWHM value to pass to photutils.detection.DAOStarFinder (in pixels)
            This is ignored if `sep` is used.
        """
        # Read SIAF -- will need to convert from sky -> det for the matching,
        # and from sky -> Idl to feed the fit
        siaf_data = siaf.RomanSiaf().read_roman_siaf()

        # TODO load ePSF and use it for detection, potentially
        if use_sep:
            # SEP is picky with endianness
            if img.dtype.byteorder != '=':
                bkg = sep.Background(img.byteswap().newbyteorder())
                objects = sep.extract(img.byteswap().newbyteorder()-bkg,
                                      nsigma_det, err=bkg.globalrms)
            else:
                bkg = sep.Background(img)
                objects = sep.extract(img-bkg, nsigma_det, err=bkg.globalrms)
            x_sci = objects['x']
            y_sci = objects['y']
            # TODO include additional selection cuts
        else:
            mean, median, std = sigma_clipped_stats(img)
            daofind = DAOStarFinder(fwhm=fwhm_det, threshold=nsigma_det*std)
            objects = daofind(img - median)
            x_sci = objects['xcentroid']
            y_sci = objects['ycentroid']

        # Populate matched catalog
        self.matched_cat = dict()
        self.matched_cat['x_sci'] = x_sci
        self.matched_cat['y_sci'] = y_sci

        # HARDCODED!! These are SCI positions (1-indexed)
        x_ref = self.refcat['col6'] - 1  # in px units
        y_ref = self.refcat['col7'] - 1
        # TODO the regular data will come as RA, Dec
        # Transform sky -> sci for the matching

        # Set up astropy SkyCoord objects for matching
        coord_det = SkyCoord(x=x_sci, y=y_sci, z=np.ones(len(x_sci)), unit=u.pixel,
                             representation_type='cartesian')
        coord_ref = SkyCoord(x=x_ref, y=y_ref, z=np.ones(len(x_ref)), unit=u.pixel,
                             representation_type='cartesian')

        idx, _, _ = match_coordinates_3d(coord_det, coord_ref)

        aperture = siaf_data[f'{detector}_FULL']
        # Transform Sci->Idl the reference catalog positions
        x_ref, y_ref = aperture.sci_to_idl(x_ref[idx], y_ref[idx])
        # TODO with the real data we will have to do sky -> Idl so
        # it does not rely as much on SIAF
        self.matched_cat['x_ref'] = x_ref  # These are Idl [arcsec]
        self.matched_cat['y_ref'] = y_ref

    def save_matches(self, output_path):
        """
        Save matched_cat

        Inputs
        ------
        output_path (string):
            Path to output matched catalog.
        """
        format = None
        if 'fits' not in output_path.split('.'):
            format = 'ascii'
        if self.matched_cat is not None:
            astropy.table.Table(self.matched_cat).write(output_path,
                                                        format=format)
        else:
            raise ValueError('The matched catalog is empty. \
                              Please run match_refcat first.')
