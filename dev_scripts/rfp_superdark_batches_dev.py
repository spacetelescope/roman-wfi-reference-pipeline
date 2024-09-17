from wfi_reference_pipeline.reference_types.dark.superdark_file_batches import SuperDarkBatches
import os
# import numpy as np
from memory_profiler import profile
import time

input_directory = "/grp/roman/RFP/DEV/sim_inflight_calplan/WFIsim_darks/asdf_files"

# Get all files in the directory
files = os.listdir(input_directory)

# Assuming short dark files contain '00444' and long dark files contain '00445'
file_list = [file for file in files if "WFI03" in file]
short_dark_file_list = [file for file in file_list if '00444' in file]
print("Short dark files ingested.")
for f in short_dark_file_list:
    print(f)
long_dark_file_list = [file for file in file_list if '00445' in file]
print("Long dark files ingested.")
for f in long_dark_file_list:
    print(f)

# Switch to run the method
run_superdark_batches = True
# Add the @profile decorator to the function we want to monitor


@profile
def run_superdark():
    print('Running superdark batches')
    rfp_superdark_batches_dev = SuperDarkBatches(input_path=input_directory,
                                                 short_dark_file_list=short_dark_file_list,
                                                 long_dark_file_list=long_dark_file_list)
    start_time = time.time()  # Start time
    rfp_superdark_batches_dev.make_superdark_with_batches(short_batch_size=4,
                                                          long_batch_size=4)
    rfp_superdark_batches_dev.generate_outfile()
    end_time = time.time()  # End time
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if run_superdark_batches:
    run_superdark()
