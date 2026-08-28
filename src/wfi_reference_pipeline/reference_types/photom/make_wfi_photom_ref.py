import os
os.environ['CRDS_SERVER_URL'] = 'https://roman-crds.stsci.edu'
os.environ['CRDS_PATH'] = '/Users/calamida/crds_cache'
os.environ['CRDS_CONTEXT'] = 'roman_0046.pmap'


import make_pam as pam
import synphot as syn
from roman_datamodels import stnode as rds   # First Roman Data Models Call
from astropy import units as u
from astropy.time import Time
import datetime
import asdf
import numpy as np
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Empirical1D
from astropy.table import Table
from astropy.io import ascii
from crds import getreferences
from astropy.stats import sigma_clipped_stats
import roman_datamodels as rdm               # Second Roman Data Models Call
import rad
print('rad version:')
rad.__version__

meta = {'ROMAN.META.INSTRUMENT.NAME': 'WFI',
        'ROMAN.META.EXPOSURE.START_TIME': '2024-01-01 00:00:00'}

stats = {}

for i in range(18):
    meta.update({'ROMAN.META.INSTRUMENT.DETECTOR': f'WFI{i+1:02d}'})
    gain = getreferences(meta, reftypes=['gain'], observatory='roman', context='roman_0046.pmap', ignore_cache=False)
    with asdf.config_context() as cfg:
        cfg.validate_on_read = False
        gfile = asdf.open(gain['gain'])
        garr = gfile['roman']['data'] #.value.copy()
    _, med, std = sigma_clipped_stats(garr, sigma=4, maxiters=3)
    stats[f'WFI{i+1:02d}'] = {'median': med, 'stddev': std}


"""
Create Roman WFI PHOTOM reference file that validates against the
wfi_img_photom.yaml schema.

- Imaging filters (e.g., F062) use wavelength + effective area from the ECSV, currently Roman_effarea_v8_SCA01_20240301.ecsv
- GRISM/PRISM get valid placeholder arrays (zeros for effective area).
- DARK entry uses all nulls.
"""

# Detector index
DET_INDEX = 11  # produces meta.instrument.detector = 'WFIDET_INDEX'
    
# Collecting area in m^2 (scalar allowed by schema)
COLLECTING_AREA_M2 = 3.60767

# Imaging filters (must exist as columns in the ECSV table)
#IMAGING_FILTERS = ['F106']  #['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']
IMAGING_FILTERS = ['F062', 'F087', 'F106', 'F129', 'F146', 'F158', 'F184', 'F213']
# Dispersers to include without curves
INCLUDE_DISPERSERS = True
DISPERSERS = ['GRISM', 'PRISM']  # 'GRISM_0' is also accepted by the schema

# Input throughput table (effective area curves)
#ECSV_PATH = '/grp/roman/calamida/photom/Roman_effarea_v8_SCA01_20240301.ecsv'  
ECSV_PATH = f'/grp/roman/calamida/photom/Roman_effarea_tables_20240327/Roman_effarea_v8_SCA{DET_INDEX:02d}_20240301.ecsv'

# Does the ECSV contain EFFECTIVE AREA (m^2) instead of throughput?
# For Roman_effarea_v8_SCA01_20240301.ecsv this is True.
ECSV_HAS_EFFECTIVE_AREA_M2 = True

# Output ASDF reference file
#OUTPUT_ASDF = '/grp/roman/calamida/photom/roman_WFI02_photom.asdf'
OUTPUT_ASDF = f'roman_wfi{DET_INDEX:02d}_photom.asdf'

# Pixel Area Map
import make_pam as pam
def get_pixel_area_sr(det_index: int) -> float:
    """
    Returns pixel area in steradians for detector `det_index` (1-based).
    Uses your pam provider.
    """
    return pam.PixelArea(det_index).get_nominal_area().to(u.sr).value

# Gain info
def get_gain_stats(det_index: int) -> dict:
    key = f"WFI{det_index:02d}"
    if key not in stats:
        raise KeyError(f"gain stats dict missing key '{key}'")
    return {
        'median': float(stats[key]['median']),
        'stddev': float(stats[key]['stddev']),
    }

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def build_filter_entry(
    waves_micron,          # Quantity [micron]
    eff_area_m2,           # Quantity [m^2]
    collecting_area_m2: float,
    det_index: int,
    gain_median: float,
    gain_stddev: float,
) -> dict:

    # Quantities used for physics/calculations only:
    wq = waves_micron.to(u.micron)          # Quantity [micron]
    eaq = eff_area_m2.to(u.m**2)            # Quantity [m^2]

    # Throughput = A_eff / A_coll (dimensionless ndarray)
    T = (eaq / (collecting_area_m2 * u.m**2)).decompose().value.astype(np.float32)

    # Build bandpass from throughput usign STsynphot function
    band = syn.SpectralElement(Empirical1D, points=wq, lookup_table=T)
    pivot = band.pivot()
    unit_resp = band.unit_response(collecting_area_m2 * u.m**2)

    pixel_area_sr = get_pixel_area_sr(det_index)

    # Convertion to MJy/sr
    mjy_per_dnps_per_sr = (
        syn.units.convert_flux(pivot, unit_resp, u.MJy) / (pixel_area_sr * u.sr)
    ).value

    # Getting Gain value for the detector
    g = float(gain_median)
    g_rerr = float(gain_stddev) / g if g != 0 else 0.0

    photmjsr = np.float32(mjy_per_dnps_per_sr * g)
    uncertainty = np.float32(mjy_per_dnps_per_sr * g * g_rerr)

    # >>> CRITICAL: store plain ndarrays (no units), float32, 1-D
    wavelength_arr     = wq.to_value(u.micron).astype(np.float32)  # ndarray
    effective_area_arr = eaq.to_value(u.m**2).astype(np.float32)   # ndarray

    
    return {
        'photmjsr': float(photmjsr),
        'uncertainty': float(uncertainty),
        'pixelareasr': float(np.float32(pixel_area_sr)),
        'collecting_area': float(np.float32(collecting_area_m2)),
        'wavelength': wavelength_arr,          # ndarray float32, 1-D
        'effective_area': effective_area_arr,  # ndarray float32, 1-D
    }

    
def build_disperser_entry(
    waves_micron,          # Quantity [micron]
    collecting_area_m2: float,
) -> dict:


    # >>> CRITICAL: convert to ndarray
    wavelength_arr = waves_micron.to_value(u.micron).astype(np.float32)  # ndarray
    effective_area_arr = np.zeros_like(wavelength_arr, dtype=np.float32) # ndarray

    return {
        'photmjsr': None,
        'uncertainty': None,
        'pixelareasr': None,
        'collecting_area': float(np.float32(collecting_area_m2)),
        'wavelength': wavelength_arr,         # ndarray float32, 1-D
        'effective_area': effective_area_arr, # ndarray float32, 1-D
    }

    

# ---------- Parameterized builder ----------

def build_photom_ref(
    det_index: int,
    output_asdf: str,
    imaging_filters: list,
    include_dispersers: bool,
    ecsv_path: str,
    collecting_area_m2: float,
    ecsv_has_effective_area_m2: bool,
):
    """
    Build one PHOTOM reference file for a single detector.
    """
    # --- Load ECSV with wavelength + curves ---
    thru_tab = ascii.read(ecsv_path)

    if 'Wave' not in thru_tab.colnames:
        raise ValueError(f"'Wave' column not found in {ecsv_path}")

    # Wavelength grid (as Quantity for calcs; will be converted to ndarray in helpers)
    waves_q = (thru_tab['Wave'].data * u.micron)

    # Detector-specific gain stats
    gstats = get_gain_stats(det_index)
    g_med = float(gstats['median'])
    g_std = float(gstats['stddev'])

    phot_table = {}



   # --- Imaging filters ---
    for se in imaging_filters:
        if se not in thru_tab.colnames:
            raise ValueError(f"Filter column '{se}' not found in {ecsv_path}")

        col = thru_tab[se].data.astype(np.float64)  # safe math
        if ecsv_has_effective_area_m2:
            eff_q = (col * u.m**2)  # Quantity
        else:
            # Throughput -> effective area
            eff_q = (col.astype(np.float32) * (collecting_area_m2 * u.m**2))

        entry = build_filter_entry(
            waves_micron=waves_q,
            eff_area_m2=eff_q,
            collecting_area_m2=collecting_area_m2,
            det_index=det_index,
            gain_median=g_med,
            gain_stddev=g_std,
        )
        phot_table[se] = entry

    # --- Dispersers ---
    if include_dispersers:
        for disp in ('GRISM', 'PRISM'):
            phot_table[disp] = build_disperser_entry(
                waves_micron=waves_q,
                collecting_area_m2=collecting_area_m2,
            )


    # --- DARK entry (all None) ---
    phot_table['DARK'] = {
        'photmjsr': None,
        'uncertainty': None,
        'pixelareasr': None,
        'collecting_area': None,
        'wavelength': None,
        'effective_area': None,
    }


    # --- Build the datamodel and meta ---
#    dm = rds.WfiImgPhotomRef()
#    dm.phot_table = phot_table


    phot_meta = {'reftype': 'PHOTOM',
        'description': 'Roman WFI absolute photometric calibration information. Throughput information comes from the Roman Technical Information Repository (https://github.com/spacetelescope/roman-technical-information) version 1.0 with a data of 2024 March 27. Gain has been accounted for to correctly transform from input count rates of DN/s to physical units. Gains have been measured as sigma-clipped (sigma=4, iterations=3) medians from the first pass TVAC1 gain reference files. Uncertainty in the zeropoint reflects the 1-sigma standard deviation in the sigma-clipped gain.', 
        'pedigree': 'GROUND',
        'telescope': 'ROMAN',
        'origin': 'STSCI/SOC',
        'author': 'A. Calamida',
        'useafter': Time(datetime.datetime(2020, 1, 1, 0, 0, 0)),
        'instrument':
            {'detector': f'WFI{DET_INDEX:02d}',
            'name': 'WFI',
            'median_gain': float(g_med),
            'sigma_gain': float(g_std)}
        }

    dm = rds.WfiImgPhotomRef()
    dm['phot_table'] = phot_table
    dm['meta'] = phot_meta



    # Optional: embed versions for reproducibility
    try:
        import roman_datamodels, astropy, synphot, asdf as _asdf
        dm.meta.software = {
            "roman_datamodels": roman_datamodels.__version__,
            "astropy": astropy.__version__,
            "numpy": np.__version__,
            "synphot": synphot.__version__,
            "asdf": _asdf.__version__,
        }
    except Exception:
        pass

    with asdf.AsdfFile() as af:
        af.tree = {'roman': dm}
        af.write_to(output_asdf)

    print(f"✓ Wrote {output_asdf}")


if __name__ == "__main__":
    # Configuration you already have in your script:
    collecting_area_m2 = COLLECTING_AREA_M2
    ecsv_path = ECSV_PATH
    ecsv_has_effective_area_m2 = ECSV_HAS_EFFECTIVE_AREA_M2
    imaging_filters = IMAGING_FILTERS
    include_dispersers = INCLUDE_DISPERSERS

    for det in range(1, 19):  # WFI01..WFI18
        output_asdf = f"roman_wfi{det:02d}_photom.asdf"
        build_photom_ref(
            det_index=det,
            output_asdf=output_asdf,
            imaging_filters=imaging_filters,
            include_dispersers=include_dispersers,
            ecsv_path=ecsv_path,
            collecting_area_m2=collecting_area_m2,
            ecsv_has_effective_area_m2=ecsv_has_effective_area_m2,
        )



