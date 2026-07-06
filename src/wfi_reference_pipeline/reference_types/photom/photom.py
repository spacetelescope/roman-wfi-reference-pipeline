import os
import shutil
import subprocess
from pathlib import Path

import crds
import numpy as np
import roman_datamodels as rdm
import synphot as syn
from astropy import units as u
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
from crds.client import api
from roman_datamodels.datamodels import WfiImgPhotomRefModel
from synphot.models import Empirical1D

from wfi_reference_pipeline.constants import (
    COLLECTING_AREA_M2,
    WFI_REF_OPTICAL_ELEMENTS,
)
from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import PixelArea
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.resources.wfi_meta_photom import (
    WFIMetaPhotom,
)

from ..reference_type import ReferenceType


class Photom(ReferenceType):
    """Class PhotomPSF() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.

    A development script can produce the files - rfp_photom_creation.py
    """

    def __init__(
        self,
        meta_data,
        outfile="roman_photom_file.asdf",
        clobber=False,
    ):
        '''
        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        outfile: str
            Output ASDF file name.
        clobber: bool
            Whether to overwrite existing ASDF file.
        '''

        super().__init__(
            meta_data=meta_data,
            outfile=outfile,
            clobber=clobber,
        )

        if not isinstance(meta_data, WFIMetaPhotom):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaPhotom"
            )
        
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI absolute photometric calibration information."


        self.meta = meta_data
        self.outfile = outfile

        # Prepare an empty didctionary to store the photom information
        self.phot_table = {}

        # Build the gain dictionary upon the initialization  
        self.gain = gain()

        # Check if the gain dictionary is properly populated
        # should be a dictionary containing the gain values for all 18 detectors
        if self.gain is not None:
            if not isinstance(self.gain, dict):
                raise ValueError('The gain must be a dictionary')
            
            if len(self.gain) != 18:
                raise ValueError('Gain dictionary does not contain information for all 18 detectors')
                

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the Photom reference file.
        """

        photom_datamodel_tree = WfiImgPhotomRefModel()
        photom_datamodel_tree['meta'] = self.meta.export_asdf_meta()
        photom_datamodel_tree['phot_table'] = self.phot_table

        return photom_datamodel_tree
    
    def calculate_error(self):
        """
        Abstract method not applicable.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass

    def _read_throughput(self):
        """
        Throughput provider. Currently only supporting the ecsv files from GSFC. 
        Future developement will include FITS tables and commissioning measurements
        """

        det = self.meta_data.instrument_detector
        det_num = str.split(det, 'WFI')[-1]     # Getting just the detector number without the prefix 'WFI'
        throughput_path = (Path(__file__).parent / 'Roman_effarea_tables_20240327' / f'Roman_effarea_v8_SCA{det_num}_20240301.ecsv').resolve()
        thru_tab = ascii.read(str(throughput_path))

        if 'Wave' not in thru_tab.colnames:
            raise ValueError(f"'Wave' column not found in {throughput_path}")
        
        # Check if the throughput table contains data for WFI optical elements
        for filter in WFI_REF_OPTICAL_ELEMENTS:
            # Re-format the filter label to match the label in the throughput table
            if filter in ['PRISM', 'GRISM']:
                if filter == 'PRISM':
                    filter = 'Prism'
                elif filter == 'GRISM':
                    filter = 'Grism_1stOrder'

            if filter not in thru_tab.colnames:
                if filter == 'DARK':  
                    # Throughput table does not contain dark information
                    pass
                else:
                    raise ValueError(f'{filter} column not found in {throughput_path}')

        return thru_tab

    def optical_element_helper(self):
        """
        Optical element helper to build entries that go into the phot_table
        The entries are:
            photomjsr: float
                Zeropoint that converts from data numbers per second (DN/s) to megaJanskys per steradian (MJy/sr).
            uncertainty: float
                Uncertainty in the zeropoint.
            pixelareasr: float32
                The nominal pixel area in units of steradians.
            collecting_area: float 
                Collecting area of the telescope.
            wavelength: 1D numpy array, float32
                Wavelength of the element in micron.
            effective_area: 1D numpy array, float32
                Effective area curve as a function of wavlenegth for the element in m^2.
        """

        throughput_table = self.throughput_table

        # Setting the optical elements as filter to manupulate the label below to match the label in the throughput 
        filter = self.meta_data.optical_element
        if filter != 'DARK':
            if filter == 'PRISM':
                filter = 'Prism'
            elif filter == 'GRISM':
                filter = 'Grism_1stOrder'
            
            col = throughput_table[filter].data.astype(np.float64)  # safe math
            waves_micron = throughput_table['Wave'].data * u.micron

            # If the input file file contains throughput and not effective area, convert the data to effective area
            # by multiplying the collecting area
            if max(throughput_table[filter]) < 1.0:
                eff_area_m2 = (col.astype(np.float32) * (COLLECTING_AREA_M2 * u.m**2))
            else:
                # This is the case where the file contains effective area
                eff_area_m2 = (col * u.m**2)


            # Quantities used for physics/calculations only:
            wq = waves_micron.to(u.micron)          # Quantity [micron]
            eaq = eff_area_m2.to(u.m**2)            # Quantity [m^2]

            # Throughput = A_eff / A_coll (dimensionless ndarray)
            thru_table = (eaq / (COLLECTING_AREA_M2 * u.m**2)).decompose().value.astype(np.float32)

            # Build bandpass from throughput usign STsynphot function
            band = syn.SpectralElement(Empirical1D, points=wq, lookup_table=thru_table)
            pivot = band.pivot()
            unit_resp = band.unit_response(COLLECTING_AREA_M2 * u.m**2)

            # PAM
            pixel_area_sr = self.pixel_area_sr
            # Convertion to MJy/sr
            mjy_per_dnps_per_sr = (
                syn.units.convert_flux(pivot, unit_resp, u.MJy) / (pixel_area_sr * u.sr)
            ).value

            # Gain
            g = self.g
            g_rerr = self.g_rerr

            photmjsr = np.float32(mjy_per_dnps_per_sr * g)
            uncertainty = np.float32(mjy_per_dnps_per_sr * g * g_rerr)

            # >>> CRITICAL: store plain ndarrays (no units), float32, 1-D
            wavelength_arr     = wq.to_value(u.micron).astype(np.float32)  # ndarray
            effective_area_arr = eaq.to_value(u.m**2).astype(np.float32)   # ndarray


        # --- Disperser elements ---
        # Currently set to null values in the photom ref file (as of 06/24/2026)
        if self.meta_data.optical_element in ['GRISM', 'PRISM']:
            return {
                'photmjsr': None,
                'uncertainty': None,
                'pixelareasr': None,
                'collecting_area': COLLECTING_AREA_M2,
                'wavelength': wavelength_arr,          # ndarray float32, 1-D
                'effective_area': effective_area_arr,  # ndarray float32, 1-D
            }
        # --- DARK entry (all None) ---
        elif self.meta_data.optical_element == 'DARK':   
            return {
                'photmjsr': None,
                'uncertainty': None,
                'pixelareasr': None,
                'collecting_area': None,                
                'wavelength': None,          # ndarray float32, 1-D
                'effective_area': None,  # ndarray float32, 1-D
            }
        # --- Imaging filters ---
        else:   
            return {
                'photmjsr': float(photmjsr),
                'uncertainty': float(uncertainty),
                'pixelareasr': float(np.float32(pixel_area_sr)),
                'collecting_area': COLLECTING_AREA_M2,
                'wavelength': wavelength_arr,          # ndarray float32, 1-D
                'effective_area': effective_area_arr,  # ndarray float32, 1-D
            }


    def make_photom(self):
        '''
        Build the photom dictionary for each detector
        '''
        
        # Instantiate an empty dictionary to store the photom table
        phot_table = {}


        # Read the throughout curve
        self.throughput_table = self._read_throughput()


        # Get the pixel area map for the detector
        rfp_pam = pam(self.meta_data.instrument_detector)
        self.pixel_area_sr = rfp_pam.meta_data.pixelarea_steradians


        # Get the gain value for the detector
        self.g = float(self.gain[self.meta_data.instrument_detector]['median'])
        self.g_rerr = float(self.gain[self.meta_data.instrument_detector]['std']) / self.g if self.g != 0 else 0.0


        print('Building the phot_table for each element')
        for optical_element in WFI_REF_OPTICAL_ELEMENTS:
            self.meta_data.optical_element = optical_element
            phot_table[optical_element] = self.optical_element_helper()

        # Save the results
        self.meta_data.median_gain = self.g
        self.meta_data.sigma_gain = self.g_rerr
        self.phot_table = phot_table




# -------------------------------
# Standalone functions for gain and pixel area map
# -------------------------------
def gain():
    """
    Grab the gain files from CRDS and compute the sigma clipped median gain

    """
    """
    Parameters
    ----------
    output_dir : str
        Path to the directory where CRDS reference files will be cached/downloaded.
    """

    output_dir = (Path(__file__).parent / 'cache').resolve()

    print("CRDS_SERVER_URL:", os.environ.get("CRDS_SERVER_URL"))
    print("CRDS_PATH:", os.environ.get("CRDS_PATH"))

    crds_context = crds.get_default_context(observatory='roman', state='latest')
    print(f"CRDS context: {crds_context}")

    if os.path.exists(output_dir):
        print(f"Deleting existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("Syncing CRDS reference files...")
    try:
        result = subprocess.run(
            ["crds", "sync", "--all"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running crds sync: {e.stderr}")

    
    # -------------------------------
    # Locally download the gain file
    # -------------------------------
    
    gain_files = crds.rmap.load_mapping(crds.get_default_context()).get_imap('wfi').get_rmap('gain').reference_names()
    results = api.dump_references(crds_context, gain_files)
    gain_filepaths = list(results.values())

    print("Building a gain dictionary for all 18 detectors")
    gain_vals = {}          # Create an empty dictionary to store the gain values for each detector
    for filepath in gain_filepaths:
        try:
            with rdm.open(filepath) as ref:
                det = ref.meta.instrument.detector
                _, med, std = sigma_clipped_stats(ref.data, sigma=4, maxiters=3)
                val = float(med)
                val_corrected = val / 1.08
                print(f"{det}: gain -> {val:.2f}, gain_corrected -> {val_corrected:.2f}")

                # Save the gain values in the dictionary to convert DNs to electrons 
                # for the rest of the detector parameters
                gain_vals[det] = {'median': val_corrected, 'std': std}
        except Exception as e:
            print(f"Error reading {filepath}: {e}")    
    
    return gain_vals


def pam(det):
    '''
    Compute the pixel area map for a specific detector
    '''

    print(f'Computing the pixal area map for {det}')
    tmp = MakeDevMeta(ref_type='AREA')    
    tmp.meta_pixelarea.instrument_detector = det
    rfp_pam = PixelArea(meta_data=tmp.meta_pixelarea)

    return rfp_pam