#import logging #uncomment if ever used

import asdf
import numpy as np
import roman_datamodels.stnode as rds
import stpsf
from photutils.aperture import CircularAperture, aperture_photometry

from wfi_reference_pipeline.constants import (
    WFI_REF_OPTICAL_ELEMENT_DARK,
    WFI_REF_OPTICAL_ELEMENT_GRISM,
    WFI_REF_OPTICAL_ELEMENT_PRISM,
    WFI_REF_OPTICAL_ELEMENTS,
)
from wfi_reference_pipeline.resources.wfi_meta_aperturecorrection import (
    WFIMetaApertureCorrection,
)

from ..reference_type import ReferenceType


class ApertureCorrection(ReferenceType):
    """
    Class ApertureCorrection() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, meta_data, outfile='roman_apcorr.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        outfile: string; default = roman_apcorr.asdf
            Filename with path for saved inverse linearity reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        """

        # Access methods of base class ReferenceType
        super().__init__(meta_data, outfile=outfile, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaApertureCorrection):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaApertureCorrection"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI aperture correction reference file."

    def calculate_aperture_corrections(self, psf_object, enclosed_energy_fractions, pixel_position=(2048,2048), oversample=21, min_radius=1, max_radius=20):
        """
        Calculate the radii of enclosed energy fractions and the associated aperture corrections.

        Generates a stpsf model and performs circular aperture photometry at integer oversampled pixel radii between min_radius and max_radius. Interpolating the aperture photometry and evaluating the interpolation at the enclosed energy fractions determines the enclosed energy pixel radii and allows for the aperture correction to be calculated.

        Parameters
        ----------
        psf_object: stpsf instrument instance
            stpsf Roman WFI instance for the correct filter and detector
        enclosed_energy_fractions : 1d-array
            the enclosed energy fractions at which to calculate the aperture corrections and the pixel radii
        pixel_position: tuple of floats; default = (2048,2048)
            the position at which to evaluate the PSF object and calculate the PSF. The default is the center of the detector as the corrections did not vary by more than 1% across the detector.
        oversample : integer; default = 21
            Oversampling rate for calculating the stpsf model. Default of 21 was the lowest value that achieved a relative error to within 1% compared to oversample = 50.
        min_radius: integer; default = 1
            Minimum radius to start the interpolation function of circular aperture photometry
        max_radius: integer; default=20
            The maximum width of the psf model in pixels over which to evaluate the aperture photometry.
        """
        # generate the oversampled PSF
        psf_object.detector_position = pixel_position
        psf_results = psf_object.calc_psf(oversample=oversample, fov_pixels=(max_radius+1))
        psf_data = psf_results[0].data

        # create a suite of circular apertures centered on the psf
        psf_center_position = tuple([(max_radius*oversample)/2]*2)
        apertures = [CircularAperture(psf_center_position, i) for i in range(min_radius, max_radius*oversample)]
        aperture_flux_table = aperture_photometry(psf_data, apertures)

        # generate lists of the fluxes and radii of each aperture
        aperture_fluxes = [aperture_flux_table[f'aperture_sum_{k}'][0]for k in range(len(apertures))]
        aperture_radii = [ap.r/oversample for ap in apertures]

        # interpolate over the enclosed energy fractions to generate a smooth function
        enclosed_energy_radii = np.interp(enclosed_energy_fractions, np.array(aperture_fluxes), np.array(aperture_radii))

        #TODO: this needs to be updated to include background effects
        aperture_corrections = 1./enclosed_energy_fractions

        # These percentiles were chosen to match the JWST definitions
        sky_in = float(np.interp(0.8, np.array(aperture_fluxes), np.array(aperture_radii)))
        sky_out = float(np.interp(0.85, np.array(aperture_fluxes), np.array(aperture_radii)))

        return enclosed_energy_radii, aperture_corrections, sky_in , sky_out

    def generate_aperture_correction_dict(self):
        """
        Create the dictionary of all required quantities needed for the APCORR reference file. The current methodology iterates over fixed enclosed energy fractions to match the JWST NIRCAM reference files. The quantities are saved under the following keys:
        - ee_fractions: enclosed energy fractions
        - ee_radii: pixel radii of each enclosed energy fraction
        - ap_corrections: aperture corrections for each enclosed energy radius
        - sky_background_rin: inner pixel radius to estimate the sky background
        - sky_background_rout: outer pixel radius to estimate the sky background
        """
        aperture_correction_dict = dict()
        # enclosed energy percentiles were chosen to match JWST
        enclosed_energy_fractions = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95])
        
        wfi = stpsf.WFI()
        wfi.detector = f'SCA{self.meta_data.instrument_detector[-2:]}'

        for optical_element in WFI_REF_OPTICAL_ELEMENTS:
            # if not an imaging mode, we do not have aperture corrections yet so return none
            if optical_element in [WFI_REF_OPTICAL_ELEMENT_DARK,
                                   WFI_REF_OPTICAL_ELEMENT_GRISM,
                                   WFI_REF_OPTICAL_ELEMENT_PRISM]:
                aperture_correction_dict[optical_element] = {
                    'ee_fractions': None,
                    'ee_radii': None,
                    'ap_corrections': None,
                    'sky_background_rin': None,
                    'sky_background_rout': None,
                }

            else:
                wfi.filter = optical_element

                enclosed_energy_radii, aperture_corrections, sky_background_inner_r, sky_background_outer_r = self.calculate_aperture_corrections(wfi, enclosed_energy_fractions)

                aperture_correction_dict[optical_element] = {
                    'ee_fractions': enclosed_energy_fractions,
                    'ee_radii': enclosed_energy_radii,
                    'ap_corrections': aperture_corrections,
                    'sky_background_rin': sky_background_inner_r,
                    'sky_background_rout': sky_background_outer_r,
                }

        return aperture_correction_dict

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        apcorr_datamodel_tree = rds.ApcorrRef()
        apcorr_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        apcorr_datamodel_tree['data'] = self.generate_aperture_correction_dict()

        return apcorr_datamodel_tree

    def save_aperture_correction(self, datamodel_tree=None):
        """
        The method save_aperture_correction writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}

        # check to see if file currently exists
        self.check_outfile()
        af.write_to(self.outfile)

    # not needed for aperture correction files
    def calculate_error(self):
        return super().calculate_error()
    def update_data_quality_array(self):
        return super().update_data_quality_array()
