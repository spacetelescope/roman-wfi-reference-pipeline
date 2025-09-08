from wfi_reference_pipeline.utilities import logging_functions
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.aperturecorrection.aperturecorrection import ApertureCorrection
from astropy.time import Time
from pathlib import Path

# configure a logging file
#logging_functions.configure_logging("wfi_aperture_correction_creation")

write_path = Path('/grp/roman/RFP/DEV/scratch/') # Set the write path to be in the RFP scratch directory.

# Start by making the generic metadata for all the files
tmp = MakeDevMeta(ref_type='APCORR')
tmp.meta_aperturecorrection.author = 'R. G. Cosentino'
tmp.meta_aperturecorrection.use_after = Time('2020-01-01T00:00:00.000')
tmp.meta_aperturecorrection.pedigree = "SIMULATION"

# Need to ensure data from STPSF is accessed so
# make sure to export STPSF_PATH=/home/'user'/data/stpsf-data/
tmp.meta_aperturecorrection.description = \
"""Aperture Correction Reference File  
Intended for romancal version: >=0.19.0  
roman_datamodels version: 0.25.0  
Software versions:  
    stpsf: 2.1.0  
    synphot: 1.5.0  

Simulated aperture corrections were generated using stpsf to simulate the PSFs and circular 
aperture photometry with photutils. Unlike JWST NIRCam, which defines corrections based on 
fixed enclosed energy fractions, this reference file uses a fixed step-size sampling of 
aperture radii in arcseconds.

Aperture corrections and enclosed energy fractions are computed for radii ranging from 0.025″ to 5.0″ 
in uniform steps of 0.025″. Radii, enclosed energy, and corrections are reported in arcseconds 
without conversion to pixel units.  

Each filter in the file contains a dictionary with:
    - ee_fractions: enclosed energy fraction at each radius (flux / total flux)
    - ee_radii: array of aperture radii in **arcseconds**
    - ap_corrections: aperture corrections (total_flux / flux_within_radius)
    - sky_background_rin: inner radius of background annulus (in arcseconds, default = 2.4″)
    - sky_background_rout: outer radius of background annulus (in arcseconds, default = 2.8″)

Only the imaging optical elements are populated. The Dark, Grism, and Prism modes are set to None.
"""

# Loop over the detectors 
for detector_index in range(1):
    detector_integer = detector_index + 1

    tmp.meta_aperturecorrection.instrument_detector = f'WFI{detector_integer:02d}'
    outfile = write_path / f'roman_dev_apcorr_{detector_integer:02d}.asdf'

    aperture_correction = ApertureCorrection(
                                meta_data=tmp.meta_aperturecorrection,
                                outfile=outfile,
                                clobber=True
                            )

    aperture_correction.save_aperture_correction() 
    print('Made file -> ', outfile)
    