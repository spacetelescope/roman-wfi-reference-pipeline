from wfi_reference_pipeline.reference_types.dark.superdark import SuperDark
import os


input_directory = "/grp/roman/RFP/DEV/sim_inflight_calplan/WFIsim_darks/asdf_files"

# Get all files in the directory
files = os.listdir(input_directory)

# Filter files that contain "WFI01" in their names
file_list_WFI01 = [file for file in files if "WFI01" in file]
# file_list_WFI01.sort()
# print("Files for WFI01.asdf:")
# for f in file_list_WFI01:
#     print(f)

# Assuming short dark files contain '00444' and long dark files contain '00445'
short_dark_file_list = [file for file in file_list_WFI01 if '00444' in file]
long_dark_file_list = [file for file in file_list_WFI01 if '00445' in file]

sd = SuperDark(input_path=input_directory,
               file_list=file_list_WFI01,
               outfile="roman_superdark.asdf")

# To do both the meta loop and the make superdark loops, set to 1
test_time_all = 1
if test_time_all == 1:
    sd.get_file_list_meta_rdmopen()
    # This above takes about 8 min and uses 5.93 GB maximum
    # Memory used in file_name loop aftering opening file: 4.49 GB
    # Total time taken to get all files meta: 501.59 seconds

    sd.make_superdark_method_c()

# To do both only the make superdark loop, set to 1 but need to provide sorted file list and number of reads list.
test_specific_methods_no_meta = 0
if test_specific_methods_no_meta == 1:
    # These arrays come from the get meta of the file list routine which takes 8 mins to run through all of the files.
    # For this testing I am putting the arrays as optional inputs into the methods to be tested to skip the meta
    # open method.
    n_reads_list_sorted = (46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                         46, 46, 46, 46, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                         98, 98, 98, 98, 98, 98)

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

    #sd.make_superdark_method_A(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)











    #sd.make_superdark_method_B(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)

    #TODO for Brad to review for testing
    # This is the benchmark method to look at for now

    sd.make_superdark_method_c(short_dark_file_list=short_dark_file_list, long_dark_file_list=long_dark_file_list)

    sd.make_superdark_method_c(n_reads_list_sorted=n_reads_list_sorted, file_list_sorted=file_list_sorted)
    sd.write_superdark(outfile='test_spuerdark.asdf')