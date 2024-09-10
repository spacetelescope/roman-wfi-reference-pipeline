from wfi_reference_pipeline.reference_types.dark.superdark import SuperDark
from wfi_reference_pipeline.reference_types.dark.superdark_dynamic import SuperDarkDynamic
import os, logging
from wfi_reference_pipeline.utilities.logging_functions import configure_logging
import cProfile
import pstats

input_directory = "/grp/roman/RFP/DEV/sim_inflight_calplan/WFIsim_darks/asdf_files"

# # Get the root logger
# logger = logging.getLogger()
# # Clear any existing handlers
# if logger.hasHandlers():
#     logger.handlers.clear()

# Get all files in the directory
files = os.listdir(input_directory)

# Filter files that contain "WFI01" in their names
file_list_WFI01 = [file for file in files if "WFI01" in file]
# file_list_WFI01.sort()
# print("Files for WFI01.asdf:")
# for f in file_list_WFI01:
#     print(f)

sd = SuperDarkDynamic(input_path=input_directory,
               file_list=file_list_WFI01,
               outfile="roman_superdark.asdf")

file_list_sorted = ('r0044401001001001001_01101_0001_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0002_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0003_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0004_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0005_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0006_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0007_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0008_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0009_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0010_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0011_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0012_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0013_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0014_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0015_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0016_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0017_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0018_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0019_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0020_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0021_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0022_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0023_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0024_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0025_WFI01_uncal.asdf',
                      'r0044401001001001001_01101_0026_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0001_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0002_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0003_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0004_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0005_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0006_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0007_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0008_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0009_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0010_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0011_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0012_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0013_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0014_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0015_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0016_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0017_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0018_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0019_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0020_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0021_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0022_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0023_WFI01_uncal.asdf',
                      'r0044501001001001001_01101_0024_WFI01_uncal.asdf')

n_reads_list_sorted = (46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                         46, 46, 46, 46, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                         98, 98, 98, 98, 98, 98)

def profile_make_superdark_dynamic():
    print("calling profile methodx")
    sd.make_superdark_dynamic(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)


# To do both the meta loop and the make superdark loops, set to 1
test_time_all = 0
if test_time_all == 1:
    sd.get_file_list_meta_rdmopen()
#     # This above takes about 8 min and uses 5.93 GB maximum
#     # Memory used in file_name loop aftering opening file: 4.49 GB
#     # Total time taken to get all files meta: 501.59 seconds

    print("CONFIGURING LOGGING")
    configure_logging("rfp_superdark_dev")
#    profile_make_superdark_dynamic()
    cProfile.run("profile_make_superdark_dynamic()", "profile_results_method_x.prof")
    p = pstats.Stats('profile_results_method_x.prof')
    p.sort_stats('cumulative').print_stats(10)

    sd.write_superdark(outfile='test_superdark.asdf')



# To do both only the make superdark loop, set to 1 but need to provide sorted file list and number of reads list.
test_specific_methods_no_meta = 1
if test_specific_methods_no_meta == 1:
    # These arrays come from the get meta of the file list routine which takes 8 mins to run through all of the files.
    # For this testing I am putting the arrays as optional inputs into the methods to be tested to skip the meta
    # open method.


    #sd.make_superdark_method_B(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)

    #TODO for Brad to review for testing
    # This is the benchmark method to look at for now

    print("CONFIGURING LOGGING")
    configure_logging("rfp_superdark_dev")
    sd.make_superdark_method_x(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)
    # cProfile.run("profile_make_superdark_dynamic()", "profile_results_method_x.prof")
    # p = pstats.Stats('profile_results_method_x.prof')
    # p.sort_stats('cumulative').print_stats(100)
    logging.debug("ABOUT TO WRITE test_superdark.asdf")
    sd.write_superdark(outfile='test_superdark.asdf')

    # This was run on the RFP dev VM
    # Method C last output. Need to get logging working
    # Memory in file loop method C: 11.21 GB
    # Memory at end of file loop in method C: 11.21 GB
    # Sigma clipping reads from all files for read
    # Memory used at end of read index loop method C: 12.61 GB
    # Read loop C time: 339.90 seconds
    # Current date and time: 2024-05-08 11:33:18.754655
    # Total time taken for method C: 41745.84 seconds

    # This was run on the RTB3 VM
    # Method C last output. Need to get logging working
    # Memory in file loop method C: 11.00 GB
    # Memory at end of file loop in method C: 11.00 GB
    # Sigma clipping reads from all files for read
    # Memory used at end of read index loop method C: 12.41 GB
    # Read loop C time: 60.34 seconds
    # Current date and time: 2024-05-09 06:08:34.775661
    # Total time taken for method C: 13489.67 seconds

    # Another un on RTB3 VM
    # Memory in file loop method C: 11.13 GB
    # Memory at end of file loop in method C: 11.13 GB
    # Sigma clipping reads from all files for read
    # Memory used at end of read index loop method C: 12.54 GB
    # Read loop C time: 60.39 seconds
    # Current date and time: 2024-05-10 10: 50:25.239326
    # Total time taken for method C: 20877.91 seconds

    # Another un on RTB3 VM
    # Memory in file loop method C: 12.54GB
    # Memory at end of file loop in method C: 12.54 GB
    # Sigma clipping reads from all files for read
    # Memory used at end of read index loop method C: 13.95 GB
    # Read loop C time: 61.92 seconds
    # Current date and time: 2024 - 05 - 14 16: 51:01.327222
    # Total time taken for method C: 17263.03 seconds


