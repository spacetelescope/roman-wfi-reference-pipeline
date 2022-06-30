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
    def __init__(self, path, detector=None):
        """
        Inputs
        ------
        path (string):
            Path to reference catalog to be read.
        detector (string):
            Name of the detector for which the distortion model is constructed.
            For example: WFI01. Optional: If provided it just selects sources in
            that detector.
        """
        # This will need to be updated
        self.refcat = astropy.table.Table.read(path, format='ascii')
        self.detector = detector
        # This will also need to be updated
        if self.detector is not None:
            self.refcat = self.refcat[self.refcat['col8'] == int(detector[-2:])]
        # Initialized empty
        self.matched_cat = None

    def match_refcat(self, detector, img, nsigma_det=5, fwhm_det=3,
                     plot_diagnostics=False):
        """
        This method runs a detection algorithm on the data contained
        in the input ReferenceFile and matches the detected sources to
        the sources contained in this catalog. After that, it populates
        the matched_cat attribute.

        Inputs
        ------
        detector (string):
            Name of the detector for which the distortion model is constructed.
            For example: WFI01. If a detector has been selected at the time of
            instantiating the reference catalog, detector should be the same.
        img (np.array):
            image to process and match to the reference catalog.
        nsigma_det (double):
            SNR threshold for detection algorithm.
        fwhm_det (double):
            FWHM value to pass to photutils.detection.DAOStarFinder (in pixels)
            This is ignored if `sep` is used.
        plot_diagnostics (bool):
            If `True` it will produce diagnostic plots for the matching routine.
            `False` by default.

        Returns
        -------
        None
        """
        # One way to avoid this is to force detector = self.detector if not None
        if (self.detector is not None) & (detector != self.detector):
            raise ValueError('Trying to match a detector different than the \
                             one selected. Please reload the reference catalog \
                             or select a different detector')
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
        sensor_sel = self.refcat['col8'] == int(detector[-2:])
        x_ref = self.refcat['col6'][sensor_sel].data - 1  # in px units
        y_ref = self.refcat['col7'][sensor_sel].data - 1
        # TODO the regular data will come as RA, Dec
        # Transform sky -> sci for the matching

        # Set up astropy SkyCoord objects for matching
        coord_det = SkyCoord(x=x_sci, y=y_sci, z=np.ones(len(x_sci)), unit=u.pixel,
                             representation_type='cartesian')
        coord_ref = SkyCoord(x=x_ref, y=y_ref, z=np.ones(len(x_ref)), unit=u.pixel,
                             representation_type='cartesian')

        # Match to the nearest neighbor
        idx, _, _ = match_coordinates_3d(coord_det, coord_ref)

        if plot_diagnostics:
            # Some of the plotting might need to be tweaked
            import matplotlib.pyplot as plt
            from scipy.stats import binned_statistic_2d
            f, ax = plt.subplots(1, 1)
            dx = x_sci - x_ref[idx]
            dy = y_sci - y_ref[idx]
            ax.hist(dx, histtype='step', label=r'$\Delta X$')
            ax.hist(dy, histtype='step', label=r'$\Delta Y$')
            ax.set_xlabel(r'$\Delta [pix]$', fontsize=16)
            ax.set_ylabel(r'Sources/bin', fontsize=16)
            ax.legend(loc='best')
            f.tight_layout()
            f.savefig('match_1d_residuals.pdf')

            def med_sclip(x):
                """
                Auxiliary routine to pass to binned_statistic_2d
                and get the clipped median.

                Inputs
                ------
                x (np.array):
                    Input array.

                Returns
                -------
                median (double):
                    sigma-clipped median of x.
                """
                return sigma_clipped_stats(x)[1]

            med, xe, ye, _ = binned_statistic_2d(x_sci, y_sci,
                                                 np.sqrt(dx**2+dy**2),
                                                 bins=30, statistic=med_sclip)

            f, ax = plt.subplots(1, 1)
            im = ax.imshow(med, origin='lower', vmax=1, extent=[xe[0], xe[-1], ye[0], ye[-1]])
            ax.set_xlabel(r'$X$ [px]', fontsize=16)
            ax.set_ylabel(r'$Y$ [px]', fontsize=16)
            plt.colorbar(im, label='Median astrometric offset [px]')
            f.tight_layout()
            f.savefig('match_2d_residuals.pdf')

            f, ax = plt.subplots(1, 1, figsize=(12, 12))
            qplot = ax.quiver(x_sci[::10], y_sci[::10], dx[::10], dy[::10], angles='xy')
            ax.quiverkey(qplot, X=0.9, Y=1.02, U=2.0, label='l=2 px',
                         labelpos='E', coordinates='axes')
            ax.set_xlabel('X [px]', fontsize=16)
            ax.set_ylabel('Y [px]', fontsize=16)
            f.tight_layout()
            f.savefig('match_residuals_quiver.pdf')

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

        Returns
        -------
        None
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
