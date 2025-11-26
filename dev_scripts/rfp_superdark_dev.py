import os
import time

from memory_profiler import profile

from wfi_reference_pipeline.reference_types.dark.superdark_dynamic import (
    SuperDarkDynamic,
)
from wfi_reference_pipeline.reference_types.dark.superdark_file_batches import (
    SuperDarkBatches,
)
from wfi_reference_pipeline.utilities.logging_functions import configure_logging

input_directory = "/PATH/TO/RFP/DEV/sim_inflight_calplan/WFIsim_darks/asdf_files"

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

# If true run batches, else run dynamic
run_superdark_batches = False


# Add the @profile decorator to the function we want to monitor


@profile
def run_superdark():
    kwargs = {}
    if run_superdark_batches:
        print('Running superdark batches')
        superdark = SuperDarkBatches(input_path=input_directory,
                                    short_dark_file_list=short_dark_file_list,
                                    long_dark_file_list=long_dark_file_list)
        kwargs = {"short_batch_size": 4, "long_batch_size": 4}
    else:
        print('Running superdark dynamic')
        superdark = SuperDarkDynamic(input_path=input_directory,
                                    short_dark_file_list=short_dark_file_list,
                                    long_dark_file_list=long_dark_file_list)

    start_time = time.time()  # Start time
    superdark.generate_superdark(**kwargs)
    superdark.generate_outfile()
    end_time = time.time()  # End time
    print(f"Time taken: {end_time - start_time:.2f} seconds")



configure_logging("rfp_superdark_dev")
run_superdark()
