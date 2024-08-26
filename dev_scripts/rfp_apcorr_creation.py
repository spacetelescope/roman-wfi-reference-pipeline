from wfi_reference_pipeline.utilities import logging_functions
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.aperturecorrection.aperturecorrection import ApertureCorrection
from astropy.time import Time


# configure a logging file
logging_functions.configure_logging("wfi_aperture_correction_creation")

write_path = '/grp/roman/wschultz/'#'/grp/roman/RFP/DEV/scratch/'  # Set the write path to be in the RFP scratch directory.

# Start by making the generic Meta data for all the files
tmp = MakeDevMeta(ref_type='APCORR') 
tmp.meta_aperturecorrection.author = 'W. C. Schultz'
tmp.meta_aperturecorrection.use_after = Time('2020-01-01T00:00:00.000')
tmp.meta_aperturecorrection.pedigree = "SIMULATION"
tmp.meta_aperturecorrection.description = \
"""Aperture Correction Reference File
Intended for romancal version: >0.16.2
roman_datamodels version: not yet implemented
Software versions: 
    webbpsf: 1.3.0
    synphot: 1.4.0 

Simulated aperture corrections made using webbpsf to simulate the PSFs and circular aperture photometry using synphot. The aperture corrections are organized by filter with each filter containing a dictionary of the enclosed energy fractions (ee_fractions), enclosed energy radii (ee_radii) in pixels, the aperture corrections (ap_corrections), and the two radii to use for the sky background annulus (sky_background_rin, and sky_background_rout).

The enclosed energy fractions are fixed at the JWST NIRCAM values:
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]. 
Each value of the enclosed energy fraction has an accompanying pixel radius and aperture correction.

The sky background radii were chosen such that rin and rout match the 80% and 85% enclosed enegy fraction radii respectively.

Only the imaging optical elements are populated. Dark, Grism, and Prism values are set to None.
"""




""

# Loop over the detectors
for detector_index in range(18):
    detector_integer = detector_index + 1

    tmp.meta_aperturecorrection.instrument_detector = f'WFI{detector_integer:02d}'
    outfile = write_path + f'roman_dev_apcorr_{detector_integer:02d}.asdf' 

    aperture_correction = ApertureCorrection(
                                meta_data=tmp.meta_aperturecorrection,
                                outfile=outfile,
                                clobber=True 
                            )

    aperture_correction.save_aperture_correction(no_datamodel=True) 
    print('Made file -> ', outfile)