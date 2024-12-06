import os
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise

# This directory has irrc corrected asdf files from TVAC1 total noise test with no light
tvac1_totalnoise_dir = '/grp/roman/GROUND_TESTS/TVAC1/ASDF_IRRCcorr/NOM_OPS/OTP00639_All_TV1a_R1_MCEB_IRRCcorr/'

# This directory has irrc corrected asdf files from TVAC1 WITH LIGHT
tvac1_NOT_dark_dir = '/grp/roman/GROUND_TESTS/TVAC1/NOM_OPS/OTP00636_Dark_TV1a_R3_MCEB_IRRCcorr/*/'

# Path to directory
directory = Path(tvac1_totalnoise_dir)

# Collect file statistics
files = list(directory.glob("*.asdf"))
total_size = sum(f.stat().st_size for f in files)
print(f"Total files: {len(files)}")
print(f"Total size: {total_size / (1024**3):.2f} GB")

# Create a defaultdict to group files by their full WFIdd identifiers
wfi_filelists = defaultdict(list)

# Loop through the file paths and extract the WFIdd identifier
for file in files:
    # Extract WFIdd by splitting on "_" and finding the relevant part
    filename = file.name
    parts = filename.split("_")
    for part in parts:
        if part.startswith("WFI") and part[3:5].isdigit():
            wfi_id = part[:5]  # Extract the full WFIdd (e.g., WFI01, WFI02)
            wfi_filelists[wfi_id].append(file)
            break

# Convert defaultdict to a regular dict (optional)
wfi_filelists = dict(wfi_filelists)

for i in range(1, 2):
    wfi_id = f'WFI{i:02}'
    file_list = wfi_filelists.get(wfi_id, [])
    output_dir = '/grp/roman/RFP/TVAC/TVAC1/rfp_readnoise_first_pass/'
    outfile = output_dir + 'roman_tvac1_readnoise_rfp_first_pass_'+wfi_id + '.asdf'
    tmp = MakeDevMeta(ref_type='READNOISE')
    tmp.meta_readnoise.use_after = '2023-08-01T00:00:00.000'
    tmp.meta_readnoise.description = 'Made from TVAC1 Total Noise test data from activity ' \
                                     'OTP00639_All_TV1a_R1_MCEB that had 1/f noise with the IRRC'
    tmp.meta_readnoise.instrument_detector = wfi_id
    rfp_tvac1_readnoise = ReadNoise(meta_data=tmp.meta_readnoise,
                                    file_list=file_list,
                                    outfile=outfile,
                                    clobber=True)

    rfp_tvac1_readnoise.make_readnoise_image()



# Make readnoise image from data cube.
# The make_readnoise_image() is a mini internal module pipeline that runs the steps necessary
# to create the readnoise image from a data cube.
rfp_readnoise2.make_readnoise_image()
# Save file.
rfp_readnoise2.generate_outfile()
print('Made reference file', rfp_readnoise2.outfile)