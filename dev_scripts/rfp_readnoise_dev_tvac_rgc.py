import os
import glob
import numpy as np
import roman_datamodels as rdm
from pathlib import Path


os.environ["CRDS_SERVER_URL"] = "https://roman-crds-tvac.stsci.edu"
os.environ['CRDS_PATH'] = '/user/sbetti/roman_tools/crds/tvac_crds/'

# This directory has irrc corrected asdf files from TVAC1 total noise test with no light
tvac1_totalnoise_dir='/grp/roman/GROUND_TESTS/TVAC1/NOM_OPS/OTP00639_All_TV1a_R1_MCEB_IRRCcorr/'

# This directory has irrc corrected asdf files from TVAC1 WITH LIGHT
loc = '/grp/roman/GROUND_TESTS/TVAC1/NOM_OPS/OTP00636_Dark_TV1a_R3_MCEB_IRRCcorr/*/'


# Path to your directory
directory = Path(tvac1_totalnoise_dir)

# Collect file statistics
files = list(directory.glob("*.asdf"))
total_size = sum(f.stat().st_size for f in files)
print(f"Total files: {len(files)}")
print(f"Total size: {total_size / (1024**3):.2f} GB")

