import numpy as np

import asdf
from astropy.modeling.models import Polynomial2D, Mapping, Shift
# from astropy.modeling import fitting  TODO- uncomment when needed
# from astropy.stats import sigma_clip  TODO - uncomment when needed
from astropy import units as u
import roman_datamodels.stnode as rds
from soc_roman_tools.siaf import siaf

from ..reference_file import ReferenceFile
# from ..utilities.reference_catalog import ReferenceCatalog  TODO - Verify Existence


class Distortion(ReferenceFile):
    """
    Class Distortion() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_distortion() creates the ASDF distortion reference file.
    """

    def __init__(self, cdt_model, meta_data, bit_mask=None, outfile=None,
                 clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_distortion.asdf'

        # Access methods of base class ReferenceFile
        super(Distortion, self).__init__(cdt_model, meta_data, bit_mask=bit_mask,
                                         clobber=clobber)

        # Update metadata with distortion file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI distortion reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'DISTORTION'
        else:
            pass

        self.meta['input_units'] = u.pixel
        self.meta['output_units'] = u.arcsec

    def make_siaf_distortion(self, detector):
        """
        The method make_siaf_distortion() generates a distortion ASDF file with
        the input data. This version uses the SIAF as input to generate the
        distortion reference file. This is the best that can be done until
        commissioning and operations.

        Inputs
        ------
        detector (string):
            Name of the detector for which the distortion model is constructed.
            For example: WFI01.

        Returns
        -------
        None
        """
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Read in the Roman SIAF. Use the default version from soc_roman_tools.
        siaf_data = siaf.RomanSiaf().read_roman_siaf()
        aperture = siaf_data[f'{detector}_FULL']

        # Find the shift between (x_sci, y_sci) = (0, 0) and the reference location.
        x_center = Shift(-aperture.XSciRef)
        y_center = Shift(-aperture.YSciRef)

        # Retrieve the distortion coefficients. We define the forward coefficients
        # to be Sci -> Idl and the inverse to be Idl -> Sci. We need both sets
        # of coefficients.
        x_for, y_for = siaf.get_distortion_coeffs(f'{detector}_FULL')
        x_inv, y_inv = siaf.get_distortion_coeffs(f'{detector}_FULL', inverse=True)

        # Retrieve V frame information.
        v3_angle = np.radians(aperture.V3IdlYAngle)
        vidl_parity = aperture.VIdlParity
        v2_ref, v3_ref = Shift(aperture.V2Ref), Shift(aperture.V3Ref)

        # Make the forward model.
        sci2idl_x = Polynomial2D(5, **x_for)
        sci2idl_y = Polynomial2D(5, **y_for)

        xc = dict()
        yc = dict()

        xc['c1_0'] = vidl_parity * np.cos(v3_angle)
        xc['c0_1'] = np.sin(v3_angle)
        yc['c1_0'] = -vidl_parity * np.sin(v3_angle)
        yc['c0_1'] = np.cos(v3_angle)
        xc['c0_0'] = 0
        yc['c0_0'] = 0

        idl2v_x = Polynomial2D(1, **xc)
        idl2v_y = Polynomial2D(1, **yc)

        # Make the inverse model.
        idl2sci_x = Polynomial2D(5, **x_inv)
        idl2sci_y = Polynomial2D(5, **y_inv)

        xc = dict()
        yc = dict()

        xc['c1_0'] = vidl_parity * np.cos(v3_angle)
        xc['c0_1'] = vidl_parity * -np.sin(v3_angle)
        yc['c1_0'] = np.sin(v3_angle)
        yc['c0_1'] = np.cos(v3_angle)
        xc['c0_0'] = 0
        yc['c0_0'] = 0

        v2idl_x = Polynomial2D(1, **xc)
        v2idl_y = Polynomial2D(1, **yc)

        # Now combine the X & Y models into a single object. Include the inverse
        # models as well.
        sci2idl = Mapping([0, 1, 0, 1]) | sci2idl_x & sci2idl_y
        sci2idl.inverse = Mapping([0, 1, 0, 1]) | idl2sci_x & idl2sci_y

        idl2v = Mapping([0, 1, 0, 1]) | idl2v_x & idl2v_y
        idl2v.inverse = Mapping([0, 1, 0, 1]) | v2idl_x & v2idl_y

        # Make the core model object.
        core_model = sci2idl | idl2v

        # Add an index shift as Python is zero-index (SIAF is one-indexed).
        index_shift = Shift(1)

        self.data = index_shift & index_shift | x_center & y_center | \
            core_model | v2_ref & v3_ref

    def make_distortion_from_stars(self, detector, img, refcat_path,
                                   degree=5, init_as_siaf=True, niter=10,
                                   **match_kwargs):
        """
        The method make_distortion_from_stars generates a distortion ASDF file with
        the input data. A lot of inspiration drawn from:
        https://github.com/spacetelescope/astrometry-scripts/blob/master/focal_plane_alignment/jwst_distortion_calibration.py
        Check this out:
        https://github.com/spacetelescope/pystortion/blob/2366d010564a8cea49534eb0af605d6374576d71/pystortion/crossmatch.py
        Docs:
        https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-001550.pdf
        Inputs
        ------
        detector (string):
            Name of the detector for which the distortion model is constructed.
            For example: WFI01.
        img (np.array):
            Array containing the calibration image used to estimate the distortion coefficients.
        refcat_path (string):
            Path to reference catalog.
        degree (int):
            Degree of the 2D polynomial to use.
        init_as_siaf (bool):
            If `True` initialize the fit with the SIAF values. If `False` initialize
            at zero.
        niter (int):
            Number of iterations to pass to FittingWithOutlierRemoval.

        Returns
        -------
        None
        """

        # TODO - NOTE!  MANY LINES OF CODE HERE ARE COMMENTED OUT BECAUSE THEY DEPEND UPON A MISSING METHOD ReferenceCatalog
        #  COMMENTING OUT TO PASS CURRENTING CI LINTING UPDATES, BUT DONT WANT TO DELETE WITHOUT CONFIRMATION OF REMOVAL

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Instantiate the reference catalog and read
        # RefCat = ReferenceCatalog(refcat_path, detector=detector) # TODO - Verify Existence

        # I probably should read/pass the image in a format that guarantees that
        # is from the detector name passed by `detector`.
        # Run detection and match the catalogs
        #RefCat.match_refcat(detector, img, **match_kwargs)

        # Get the dictionary/table
        #refcat = RefCat.matched_cat

        # For now rely on alignment data from siaf (this will probably need an update?)
        siaf_data = siaf.RomanSiaf().read_roman_siaf()
        aperture = siaf_data[f'{detector}_FULL']

        # Find the shift between (x_sci, y_sci) = (0, 0) and the reference location.
        x_center = Shift(-aperture.XSciRef)
        y_center = Shift(-aperture.YSciRef)

        # Create models, we can initialize at 0 or at the SIAF values.
        if init_as_siaf:
            x_for, y_for = siaf.get_distortion_coeffs(f'{detector}_FULL')
            x_inv, y_inv = siaf.get_distortion_coeffs(f'{detector}_FULL', inverse=True)
            # Models use initial models from SIAF
            sci2idl_x = Polynomial2D(degree=degree, **x_for)
            sci2idl_y = Polynomial2D(degree=degree, **y_for)
            idl2sci_x = Polynomial2D(degree=degree, **x_inv)
            idl2sci_y = Polynomial2D(degree=degree, **y_inv)
        else:
            # Empty initialization
            sci2idl_x = Polynomial2D(degree=degree)
            sci2idl_y = Polynomial2D(degree=degree)
            idl2sci_x = Polynomial2D(degree=degree)
            idl2sci_y = Polynomial2D(degree=degree)

        # Initialize fitter (it complains if I use LevMarLSQFitter)
        #lsqfitter = fitting.LinearLSQFitter()

        # We are going to use outlier removal
        #fit = fitting.FittingWithOutlierRemoval(lsqfitter, sigma_clip, niter=niter)

        # Get sci2idl and idl2sci transforms
        # sci2idl_x, _ = fit(sci2idl_x, refcat['x_sci'], refcat['y_sci'], refcat['x_ref'])
        # sci2idl_y, _ = fit(sci2idl_y, refcat['x_sci'], refcat['y_sci'], refcat['y_ref'])
        # idl2sci_x, _ = fit(idl2sci_x, refcat['x_ref'], refcat['y_ref'], refcat['x_sci'])
        # idl2sci_y, _ = fit(idl2sci_y, refcat['x_ref'], refcat['y_ref'], refcat['y_sci'])

        # Combine both axes
        sci2idl = Mapping([0, 1, 0, 1]) | sci2idl_x & sci2idl_y
        sci2idl.inverse = Mapping([0, 1, 0, 1]) | idl2sci_x & idl2sci_y

        # Retrieve V frame information.
        v3_angle = np.radians(aperture.V3IdlYAngle)
        vidl_parity = aperture.VIdlParity
        v2_ref, v3_ref = Shift(aperture.V2Ref), Shift(aperture.V3Ref)

        xc = dict()
        yc = dict()

        xc['c1_0'] = vidl_parity * np.cos(v3_angle)
        xc['c0_1'] = np.sin(v3_angle)
        yc['c1_0'] = -vidl_parity * np.sin(v3_angle)
        yc['c0_1'] = np.cos(v3_angle)
        xc['c0_0'] = 0
        yc['c0_0'] = 0

        idl2v_x = Polynomial2D(1, **xc)
        idl2v_y = Polynomial2D(1, **yc)

        xc = dict()
        yc = dict()

        xc['c1_0'] = vidl_parity * np.cos(v3_angle)
        xc['c0_1'] = vidl_parity * -np.sin(v3_angle)
        yc['c1_0'] = np.sin(v3_angle)
        yc['c0_1'] = np.cos(v3_angle)
        xc['c0_0'] = 0
        yc['c0_0'] = 0

        v2idl_x = Polynomial2D(1, **xc)
        v2idl_y = Polynomial2D(1, **yc)

        idl2v = Mapping([0, 1, 0, 1]) | idl2v_x & idl2v_y
        idl2v.inverse = Mapping([0, 1, 0, 1]) | v2idl_x & v2idl_y

        # Make the core model object.
        core_model = sci2idl | idl2v

        # Add an index shift as Python is zero-index (SIAF is one-indexed).
        index_shift = Shift(1)

        self.data = index_shift & index_shift | x_center & y_center | \
            core_model | v2_ref & v3_ref

    def save_file(self):
        distortion_file = rds.DistortionRef()
        distortion_file['coordinate_distortion_transform'] = self.data
        distortion_file['meta'] = self.meta
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': distortion_file}
        af.write_to(self.outfile)
