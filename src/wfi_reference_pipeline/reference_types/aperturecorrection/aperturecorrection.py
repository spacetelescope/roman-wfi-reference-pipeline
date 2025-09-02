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
    
    def calculate_aperture_corrections_arcsec(self,
                                            psf_object,
                                            radii_arcsec=np.arange(0.025, 5.01, 0.025),
                                            oversample=21,
                                            pixel_position=(2048, 2048),
                                            sky_annulus_arcsec=(0.88, 1.1),
                                            pixel_scale=0.11,
                                            ):
        """
        Calculate the aperture corrections and enclosed energy fractions
        using radii defined in arcseconds. The pixel scale is only used internally
        for oversamlping.

        Parameters
        ----------
        psf_object: stpsf instrument instance
            Roman WFI PSF model instance.
        radii_arcsec : ndarray
            Aperture radii in arcseconds.
        oversample : int
            Oversampling factor for PSF generation.
        pixel_position : tuple
            Detector position to evaluate PSF.
        sky_annulus_arcsec : tuple
            Inner and outer radius for sky background annulus (arcsec).
        pixel_scale : float
            Detector scale in arcsec/pixel (only used internally to map arcsec to pixels).
        """

        psf_object.detector_position = pixel_position

        # maximum radius in pixels for PSF stamp
        max_radius_pix = int(np.ceil(radii_arcsec.max() / pixel_scale)) + 1
        fov = max_radius_pix + 5
        psf_results = psf_object.calc_psf(oversample=oversample, fov_pixels=fov)
        psf_data = psf_results[0].data

        psf_center = tuple([(fov * oversample) / 2] * 2)

        # Convert arcsec radii → oversampled pixel radii
        radii_pix_oversampled = radii_arcsec / pixel_scale * oversample
        apertures = [CircularAperture(psf_center, r) for r in radii_pix_oversampled]
        flux_table = aperture_photometry(psf_data, apertures)
        aperture_fluxes = np.array([flux_table[f'aperture_sum_{i}'][0] for i in range(len(apertures))])

        # Use the largest aperture as total flux
        total_flux = aperture_fluxes[-1]

        # Enclosed energy fractions as a fraction of the total flux
        ee_fractions = aperture_fluxes / total_flux

        # Aperture corrections are inverse of EE fractions
        aperture_corrections = total_flux / aperture_fluxes

        # Sky annulus in arcsec
        sky_rin_arcsec, sky_rout_arcsec = sky_annulus_arcsec

        return radii_arcsec, ee_fractions, aperture_corrections, sky_rin_arcsec, sky_rout_arcsec
    
    def generate_aperture_correction_dict(self):
        """
        Create dictionary of aperture corrections and enclosed energy fractions,
        keyed by optical element. All radii are in arcseconds.

        The quantities are saved under the following keys:
        - ee_fractions : enclosed energy fraction for each aperture radius
        - ee_radii : aperture radii in arcseconds
        - ap_corrections : aperture corrections at each radius
        - sky_background_rin : inner radius of sky annulus (arcsec)
        - sky_background_rout : outer radius of sky annulus (arcsec)
        """
        aperture_correction_dict = dict()

        # PSF model
        wfi = stpsf.WFI()
        wfi.detector = f'SCA{self.meta_data.instrument_detector[-2:]}'

        # Fixed arcsecond radii range
        radii_arcsec = np.arange(0.025, 5.01, 0.025)

        for optical_element in WFI_REF_OPTICAL_ELEMENTS:
            if optical_element in [
                WFI_REF_OPTICAL_ELEMENT_DARK,
                WFI_REF_OPTICAL_ELEMENT_GRISM,
                WFI_REF_OPTICAL_ELEMENT_PRISM,
            ]:
                # Dont do for DARK, GRISM or PRISM
                aperture_correction_dict[optical_element] = {
                    'ee_fractions': None,
                    'ee_radii': None,
                    'ap_corrections': None,
                    'sky_background_rin': None,
                    'sky_background_rout': None,
                }

            else:
                wfi.filter = optical_element

                aperture_radii_arcsec, ee_fractions, aperture_corrections, sky_rin_arcsec, sky_rout_arcsec = (
                    self.calculate_aperture_corrections_arcsec(
                        psf_object=wfi,
                        radii_arcsec=radii_arcsec,
                        oversample=21,
                        pixel_position=(2048, 2048),
                        sky_annulus_arcsec=(0.88, 1.1),
                        pixel_scale=0.11,  # still needed internally
                    )
                )

                aperture_correction_dict[optical_element] = {
                    'ee_fractions': ee_fractions,
                    'ee_radii': aperture_radii_arcsec,
                    'ap_corrections': aperture_corrections,
                    'sky_background_rin': sky_rin_arcsec,
                    'sky_background_rout': sky_rout_arcsec,
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
