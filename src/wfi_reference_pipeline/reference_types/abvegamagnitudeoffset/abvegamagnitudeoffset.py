#import logging #uncomment if ever used

import asdf
import numpy as np
from astropy import units as u
import roman_datamodels.stnode as rds
from astropy.io import ascii
from pathlib import Path
from synphot import SpectralElement
from synphot.models import Empirical1D
from synphot import SourceSpectrum
from synphot.observation import Observation


from wfi_reference_pipeline.resources.wfi_meta_abvegamagnitudeoffset import WFIMetaABVegaMagnitudeOffset
import wfi_reference_pipeline.resources.data as rfp_data_module
from wfi_reference_pipeline.constants import (
    WFI_REF_OPTICAL_ELEMENTS, 
    WFI_REF_OPTICAL_ELEMENT_DARK,
    WFI_REF_OPTICAL_ELEMENT_GRISM,
    WFI_REF_OPTICAL_ELEMENT_PRISM
)

from ..reference_type import ReferenceType

class ABVegaMagnitudeOffset(ReferenceType):
    """
    Class ABVegaMagnitudeOffset() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, meta_data, outfile='roman_abvegaoffset.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        base class.

        Parameters
        ----------
       meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        outfile: string; default = roman_abvegaoffset.asdf
            Filename with path for saved inverse linearity reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        """

        # Access methods of base class ReferenceType
        super().__init__(meta_data, outfile=outfile, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaABVegaMagnitudeOffset):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaABVegaMagnitudeOffset"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI AB Vega magnitude offset reference file."

    def read_effective_area_table(self, detector_str):
        """Reads in the effective area tables for all the optical elements provided by the following linked file as of 8/20/2024. (https://roman.gsfc.nasa.gov/images/wfitech/Roman_effarea_tables_20240327.zip)
        
        **NOTE** these files include both the filter bandpasses and the detector quantum efficiencies. This may need to be changes slightly in the future to not include the quantum efficiencies.

        Parameters
        ----------
        detector_str: string
            string containing the detector number. The last two characters must be the integer (e.g. 'WFI01', 'SCA12', '05')
        """
        detector_int = int(detector_str[-2:])
        rfp_data_dir = Path(rfp_data_module.__file__).parent
        table_file_path =  rfp_data_dir / f"Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{detector_int:02d}_20240301.ecsv"
        table = ascii.read(table_file_path)
        return table

    def calculate_ab_vega_mag_offset(self, optical_element):
        """
        Calculate the correct bandpass for the provided optical element using the already calculated Vega spectrum and effective area table.

        Parameters
        ----------
        optical_element : string
            the name of the filter for which the magnitude offset should be calculated
        """
        # Calculate throughput by dividing the effective areas by the total mirror area
        mirror_area = (np.pi * (2.4/2)**2.)
        throughput = self.effective_area_table[optical_element] / mirror_area

        # generate a snyphot spectal element (need to convert angstroms to microns, hence the factor of 1e4)
        filter = SpectralElement(Empirical1D, points=self.effective_area_table['Wave']*1e4, lookup_table=throughput)
        
        # make a mock observation of Vega
        vega_observation = Observation(self.vega_spectrum, filter)

        # calculate the AB magnitude of the desired bandpass
        ab_vega_offset = vega_observation.effstim(u.ABmag)

        # because the bandpass magnitude of Vega in the vega-mag system is log(1) = 0 by definition, the AB magnitude of Vega is the same as the AB-Vega offset
        return ab_vega_offset.value

    def generate_abvega_offset_dict(self):
        """
        Creates the necessary dictionary structure to create the ABVEGAOFFSET reference file by calculating the required magnitude offsets.
        """
        abvega_offset_dict = dict()

        self.effective_area_table = self.read_effective_area_table(self.meta_data.instrument_detector)

        self.vega_spectrum = SourceSpectrum.from_vega()

        for optical_element in WFI_REF_OPTICAL_ELEMENTS:
            # if not an imaging mode, we do not have aperture corrections yet so return none
            if optical_element in [WFI_REF_OPTICAL_ELEMENT_DARK,
                                   WFI_REF_OPTICAL_ELEMENT_GRISM,
                                   WFI_REF_OPTICAL_ELEMENT_PRISM]:
                 abvega_offset_dict[optical_element] = {
                    'abvega_offset': None,
                }
            else:
                ab_vega_mag_offset = self.calculate_ab_vega_mag_offset(optical_element)
                abvega_offset_dict[optical_element] = {
                    'abvega_offset': ab_vega_mag_offset,
                }

        return abvega_offset_dict

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        abvegaoffset_datamodel_tree = rds.AbvegaoffsetRef()
        abvegaoffset_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        abvegaoffset_datamodel_tree['data'] = self.generate_abvega_offset_dict()

        return abvegaoffset_datamodel_tree

    def save_abvega_offset(self, datamodel_tree=None):
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

    # not needed for ab-vega magnitude offset files
    def calculate_error(self):
        return super().calculate_error()
    def update_data_quality_array(self):
        return super().update_data_quality_array()
