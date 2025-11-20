from pathlib import Path

from astropy.time import Time

from wfi_reference_pipeline.reference_types.abvegamagnitudeoffset.abvegamagnitudeoffset import (
    ABVegaMagnitudeOffset,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import logging_functions

# configure a logging file
logging_functions.configure_logging("wfi_ABVega_mag_offset_creation")

write_path = Path('/PATH/TO/scratch/') # Set the write path to be in the RFP scratch directory.

# Start by making the generic Meta data for all the files
tmp = MakeDevMeta(ref_type='ABVEGAOFFSET')
tmp.meta_abvegaoffset.author = 'Author Name'
tmp.meta_abvegaoffset.use_after = Time('2020-01-01T00:00:00.000')
tmp.meta_abvegaoffset.pedigree = "GROUND"
#TODO update the version numbers below to match the software used, look into automatic version number updates
tmp.meta_abvegaoffset.description = \
"""AB-Vega Magnitude Offset Reference File
Intended for romancal version: >0.17.0
roman_datamodels version: 0.23.1
Software versions:
    synphot: 1.4.0

Simulated AB-to-Vega magnitude offset made using the effective area fraction data from Goddard (https://roman.gsfc.nasa.gov/images/wfitech/Roman_effarea_tables_20240327.zip, accessed 8/20/2024). The offsets are organized by filter with each filter containing a dictionary with the magnitude offset as the sole item (abvega_offset).

The effective area fraction data is divided by the mirror area to estimate the throughput of each filter bandpass. The throuputs are then applied to the default spectrum of Vega. Calculating the AB magnitude of the bandpass applied to the Vega spectrum results in the magnitude offset.

Only the imaging optical elements are populated. Dark, Grism, and Prism values are set to None.
"""

# Loop over the detectors
for detector_index in range(18):
    detector_integer = detector_index + 1

    tmp.meta_abvegaoffset.instrument_detector = f'WFI{detector_integer:02d}'
    outfile = write_path / f'roman_dev_abvegaoffset_{detector_integer:02d}.asdf'

    abvega_offset = ABVegaMagnitudeOffset(
                                meta_data=tmp.meta_abvegaoffset,
                                outfile=outfile,
                                clobber=True
                            )

    abvega_offset.save_abvega_offset()
    print('Made file -> ', outfile)