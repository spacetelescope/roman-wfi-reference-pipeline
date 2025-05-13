import argparse
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import vstack
import galsim
import numpy as np
from romanisim import gaia, bandpass, catalog, log, wcs, persistence, parameters, ris_make_utils as ris
import s3fs

 # the command line call below
 # romanisim-make-image --nobj 0 --sca 1 --level 1 --ma_table_number 17 --truncate 2 output_image.asdf

import subprocess

command = [
    "romanisim-make-image",
    "--nobj", "0",
    "--sca", "1",
    "--level", "1",
    "--ma_table_number", "17",
    "--truncate", "2",
    "output_image.asdf"
]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Optional: print the output or handle errors
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

# Check for errors
if result.returncode != 0:
    raise RuntimeError(f"Command failed with code {result.returncode}")


"""
This notebook has how to run from python 
https://github.com/spacetelescope/roman_notebooks/blob/main/content/notebooks/romanisim/romanisim.ipynb
"""

