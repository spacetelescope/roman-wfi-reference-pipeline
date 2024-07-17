from wfi_reference_pipeline.reference_types.dark.superdark import SuperDark
import os
import numpy as np

input_directory = "/grp/roman/RFP/DEV/sim_inflight_calplan/WFIsim_darks/asdf_files"

# Get all files in the directory
files = os.listdir(input_directory)

# To do both only the make superdark loop, set to 1 but need to provide sorted file list and number of reads list.
test_method_c = 0
if test_method_c == 1:
    # These arrays come from the get meta of the file list routine which takes 8 mins to run through all of the files.
    # For this testing I am putting the arrays as optional inputs into the methods to be tested to skip the meta
    # open method.
    n_reads_list = (46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                    46, 46, 46, 46, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
                    98, 98, 98, 98, 98, 98)
    file_list = [file for file in files if "WFI03" in file]
    print("Files ingested.")
    for f in file_list:
        print(f)
    print('Testing method c')
    sd = SuperDark(input_path=input_directory,
                   file_list=file_list,
                   n_reads_list=n_reads_list,
                   outfile="roman_superdark_method_c_asdf_VMrtb2.asdf")
    sd.make_superdark_method_c(open_type='asdf')
    sd.generate_outfile()

# To do both only the make superdark loop, set to 1 but need to provide sorted file list and number of reads list.
test_specific_method_d = 0
if test_specific_method_d == 1:
    # Assuming short dark files contain '00444' and long dark files contain '00445'
    file_list = [file for file in files if "WFI03" in file]
    print("Files ingested.")
    for f in file_list:
        print(f)
    short_dark_file_list = [file for file in file_list if '00444' in file]
    for f in short_dark_file_list:
        print(f)
    long_dark_file_list = [file for file in file_list if '00445' in file]
    for f in long_dark_file_list:
        print(f)
    print('Testing method d')
    sd = SuperDark(input_path=input_directory,
                   short_dark_file_list=short_dark_file_list,
                   long_dark_file_list=long_dark_file_list,
                   outfile="roman_superdark_method_d_VMrtb3.asdf")
    sd.make_superdark_method_d()
    sd.generate_outfile()

# To do both only the make superdark loop, set to 1 but need to provide sorted file list and number of reads list.
test_specific_method_e = 1
if test_specific_method_e == 1:
    # Assuming short dark files contain '00444' and long dark files contain '00445'
    file_list = [file for file in files if "WFI01" in file]
    print("Files ingested.")
    for f in file_list:
        print(f)
    short_dark_file_list = [file for file in file_list if '00444' in file]
    for f in short_dark_file_list:
        print(f)
    long_dark_file_list = [file for file in file_list if '00445' in file]
    for f in long_dark_file_list:
        print(f)
    print('Testing method e')
    sd = SuperDark(input_path=input_directory,
                   short_dark_file_list=short_dark_file_list,
                   long_dark_file_list=long_dark_file_list)
    sd.make_superdark_method_e(short_batch_size=3,
                               long_batch_size=3)
    #sd.generate_outfile()

