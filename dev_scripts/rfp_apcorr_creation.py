from pathlib import Path

from astropy.time import Time

from wfi_reference_pipeline.reference_types.aperturecorrection.aperturecorrection import (
    ApertureCorrection,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import logging_functions

# configure a logging file
logging_functions.configure_logging("wfi_aperture_correction_creation")

write_path = Path('/PATH/TO/scratch/') # Set the write path to be in the RFP scratch directory.

# Start by making the generic Meta data for all the files
tmp = MakeDevMeta(ref_type='APCORR')
tmp.meta_aperturecorrection.author = 'Richard Cosentino'
tmp.meta_aperturecorrection.use_after = Time('2020-01-01T00:00:00.000')
tmp.meta_aperturecorrection.pedigree = "SIMULATION"
tmp.meta_aperturecorrection.description = \
"""Aperture Correction Reference File
Intended for romancal version: >0.22.0
roman_datamodels version: 0.31.0
stpsf: 2.2

Simulated aperture corrections are generated using stpsf to model PSFs and compute encircled energy (EE) profiles directly via stpsf.measure_ee. The aperture corrections are organized by filter, with each filter containing a dictionary of:

ee_radii: aperture radii in arcseconds, defined on a fixed grid from 0.025 to 5.0 arcsec in steps of 0.025
ee_fractions: enclosed energy fraction evaluated at each radius
ap_corrections: aperture corrections at each radius, computed as the inverse of the EE fraction (1 / EE)
sky_background_rin, sky_background_rout: placeholders for sky annulus radii (currently not used and set to None)

The enclosed energy fractions are not fixed in advance. Instead, EE is evaluated as a continuous function of radius and sampled at predefined aperture radii. This removes the need for explicit aperture photometry and background subtraction, since the PSFs are noiseless simulations.

Only imaging optical elements are populated. Dark, Grism, and Prism entries are set to None.
"""

# Loop over the detectors
for detector_index in range(18):
    detector_integer = detector_index + 1

    tmp.meta_aperturecorrection.instrument_detector = f'WFI{detector_integer:02d}'
    outfile = f'roman_apcorr_WFI{detector_integer:02d}.asdf'

    aperture_correction = ApertureCorrection(
                                meta_data=tmp.meta_aperturecorrection,
                                outfile=outfile,
                                clobber=True
                            )

    aperture_correction.generate_outfile()
    print('Made file -> ', outfile)
