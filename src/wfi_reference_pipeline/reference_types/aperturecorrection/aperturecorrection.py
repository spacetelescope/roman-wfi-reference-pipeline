#import logging #uncomment if ever used

import numpy as np
import roman_datamodels.stnode as rds
import stpsf

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

'''
***************************************************************************
***** NOTE ON INSTALLING STPSF ON VIRTUAL MACHINES AND DATA DIRECTORY *****
***************************************************************************

When pip installing stpsf, the necessary data files will be installed automatically in the default path, 
$HOME/data/stpsf-data. The .bash_profile needs to be updated to include the environment variable
export STPSF_PATH="$HOME/data/stpsf-data/" also. 
'''


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
                                              oversample=21,
                                              pixel_position=(2048, 2048),
                                              ):
        """
        Compute EE fractions and aperture corrections for radii from
        0.025 to 5.0 arcsec in steps of 0.025 using stpsf.measure_ee.
        """

        # Define radii grid
        radii_arcsec = np.arange(0.025, 5.0001, 0.025)

        # Configure PSF
        psf_object.detector_position = pixel_position

        psf = psf_object.calc_psf(oversample=oversample,
                                  fov_arcsec=radii_arcsec.max() * 2
                                  )

        # Get EE interpolator
        ee_interp = stpsf.measure_ee(psf,
                                     ext='OVERSAMP',
                                     binsize=psf_object.pixelscale / oversample
                                     )

        # Evaluate EE at each radius
        ee_fractions = np.array([ee_interp(r) for r in radii_arcsec])

        # Aperture corrections
        aperture_corrections = 1.0 / ee_fractions

        return radii_arcsec, ee_fractions, aperture_corrections

    def generate_aperture_correction_dict(self):
        """
        Create the dictionary of aperture correction quantities using
        a fixed grid of radii (arcsec).
        """

        aperture_correction_dict = dict()

        wfi = stpsf.roman.WFI()

        for optical_element in WFI_REF_OPTICAL_ELEMENTS:

            if optical_element in [
                WFI_REF_OPTICAL_ELEMENT_DARK,
                WFI_REF_OPTICAL_ELEMENT_GRISM,
                WFI_REF_OPTICAL_ELEMENT_PRISM
            ]:
                aperture_correction_dict[optical_element] = {
                    'ee_fractions': None,
                    'ee_radii': None,
                    'ap_corrections': None,
                    'sky_background_rin': None,
                    'sky_background_rout': None,
                }
                continue

            wfi.detector = f'SCA{self.meta_data.instrument_detector[-2:]}'
            wfi.filter = optical_element

            radii_arcsec, ee_fractions, aperture_corrections = self.calculate_aperture_corrections_arcsec(wfi)

            aperture_correction_dict[optical_element] = {
                'ee_fractions': ee_fractions,
                'ee_radii': radii_arcsec,
                'ap_corrections': aperture_corrections,
                'sky_background_rin': 2.4,
                'sky_background_rout': 2.8,
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

    # not needed for aperture correction files
    def calculate_error(self):
        return super().calculate_error()
    def update_data_quality_array(self):
        return super().update_data_quality_array()
