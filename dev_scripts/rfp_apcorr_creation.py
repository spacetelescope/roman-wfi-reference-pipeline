from wfi_reference_pipeline.utilities import logging_functions
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.aperturecorrection.aperturecorrection import ApertureCorrection
from astropy.time import Time
from pathlib import Path

# configure a logging file
#logging_functions.configure_logging("wfi_aperture_correction_creation")

write_path = Path('/grp/roman/RFP/DEV/scratch/') # Set the write path to be in the RFP scratch directory.

# Start by making the generic Meta data for all the files
tmp = MakeDevMeta(ref_type='APCORR')
tmp.meta_aperturecorrection.author = 'R. G. Cosentino'
tmp.meta_aperturecorrection.use_after = Time('2020-01-01T00:00:00.000')
tmp.meta_aperturecorrection.pedigree = "SIMULATION"
#TODO update the version numbers below to match the software used, look into automatic version number updates
#TODO update webbpsf to stpsf going forward
# tmp.meta_aperturecorrection.description = \
# """Aperture Correction Reference File
# Intended for romancal version: >0.18.0  
# roman_datamodels version: 0.23.1
# Software versions: 
#     webbpsf: 1.3.0
#     synphot: 1.4.0 

# Simulated aperture corrections made using stpsf to simulate the PSFs and circular aperture photometry using synphot. The aperture corrections are organized by filter with each filter containing a dictionary of the enclosed energy fractions (ee_fractions), enclosed energy radii (ee_radii) in pixels, the aperture corrections (ap_corrections), and the two radii to use for the sky background annulus (sky_background_rin, and sky_background_rout).

# The enclosed energy fractions are fixed at the JWST NIRCAM values:
#     x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]. 
# Each value of the enclosed energy fraction has an accompanying pixel radius and aperture correction.

# The sky background radii were chosen such that rin and rout match the 80% and 85% enclosed enegy fraction radii respectively.

# Only the imaging optical elements are populated. Dark, Grism, and Prism values are set to None.
# """

# new description 
tmp.meta_aperturecorrection.description = \
"""Aperture Correction Reference File  
Intended for romancal version: >=0.19.0  
roman_datamodels version: 0.25.0  
Software versions:  
    stpsf: 2.1.0  
    synphot: 1.5.0  

Simulated aperture corrections were generated using stpsf to simulate the PSFs and circular aperture photometry with photutils. Instead of computing aperture corrections based on fixed enclosed energy fractions (as in JWST NIRCAM), this reference file uses a fixed step-size sampling of aperture radii in arcseconds.

Aperture corrections are computed for radii ranging from 0.0" to 5.0" in uniform steps of 0.025". Due to limitations in photutils (which requires radii to be positive), a minimum radius of 0.001" is substituted for 0.0" in the photometry, while maintaining the uniform step spacing logic. Aperture radii are stored in arcseconds but also converted to pixel units using a pixel scale of 0.11 arcsec/pixel.

Each filter in the file contains a dictionary with:
    - ee_fractions: None (not used in this method)
    - ee_radii: array of aperture radii in **pixels**
    - ap_corrections: aperture corrections (flux ratios relative to total flux)
    - sky_background_rin: inner radius (in pixels) for background annulus (default = 0.88")
    - sky_background_rout: outer radius (in pixels) for background annulus (default = 1.1")

Only the imaging optical elements are populated. The Dark, Grism, and Prism modes are set to None, as aperture corrections are not currently computed for these.
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