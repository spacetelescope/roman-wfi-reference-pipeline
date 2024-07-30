from wfi_reference_pipeline.reference_types.dark.superdark import SuperDark
import os
import numpy as np
from memory_profiler import profile

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
    file_list = [file for file in files if "WFI03" in file]
    print("Files ingested.")
    #for f in file_list:
        #print(f)
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
    sd.make_superdark_method_e(short_batch_size=8,
                               long_batch_size=8)
    #sd.generate_outfile()



import matplotlib.pyplot as plt

# Data from the results
log_files = [
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b2.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b3.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b4.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b5.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_01_b6.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_02_b2.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_02_b3.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_02_b4.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_03_b2.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_03_b4.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_04_b2.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw4-1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw5-1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw6-1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw7-1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw8-1.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw8-2.log",
    "/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/dev_scripts/rgc_superdark_method_e_VMdevRFP_bmw8-3.log",
]

total_times = [
    4055.73, 3607.74, 3601.79, 3628.5, 3700.01, 3693.02, 3737.12, 3789.72, 4211.05,
    4234.62, 3728.14, 4264.1, 3783.17, 3753.49, 3541.29, 3553.15, 3573.76, 3878.26,
    3681.18
]

min_read_time_0_45 = [
    54.77, 49.54, 49.54, 49.44, 50.85, 49.63, 51.46, 49.97, 56.69, 54.52, 49.61,
    54.4, 42.93, 42.87, 42.97, 43.88, 42.95, 47.43, 47.14
]

max_read_time_0_45 = [
    62.98, 53.54, 63.67, 55.3, 55.82, 62.78, 53.2, 73.32, 76.77, 63.74, 56.15,
    69.69, 62.51, 55.59, 51.77, 52.22, 52.07, 77.06, 61.56
]

avg_read_time_0_45 = [
    57.279555555555554, 50.54288888888889, 50.489333333333335, 50.92688888888889,
    51.53111111111111, 51.66155555555556, 52.166, 53.62577777777778, 60.14333333333333,
    58.47977777777778, 52.54222222222222, 60.17955555555555, 52.04511111111111,
    50.398, 49.18311111111111, 49.37711111111111, 49.80155555555555, 56.196444444444445,
    51.37222222222222
]

min_read_time_46_98 = [
    26.59, 24.27, 24.16, 24.11, 24.75, 24.46, 25.19, 24.8, 26.81, 28.09, 23.85,
    26.14, 25.01, 25.13, 24.05, 23.76, 23.97, 24.16, 24.61
]

max_read_time_46_98 = [
    55.47, 50.11, 49.93, 49.85, 51.56, 49.9, 51.69, 51.86, 58.41, 58.41, 53.59,
    58.65, 55.05, 53.47, 50.13, 49.86, 49.22, 49.87, 52.19
]

avg_read_time_46_98 = [
    27.88924528301887, 25.156981132075472, 25.09075471698113, 25.221509433962265,
    26.057735849056602, 25.81641509433962, 26.21981132075472, 25.97320754716981,
    28.38811320754717, 30.245471698113207, 25.730943396226415, 29.358679245283017,
    27.19132075471698, 28.02943396226415, 25.05735849056604, 25.11622641509434,
    25.144528301886794, 25.461509433962263, 25.838679245283018
]

# Extract batch sizes from log file names
batch_labels = []
for log_file in log_files:
    if 'bmw' in log_file:
        batch_size = log_file.split('_')[-1].split('-')[0][3:]
        label = f'bmw {batch_size}'
    else:
        batch_size = log_file.split('_')[-1][1:].split('.')[0]
        label = f'b {batch_size}'
    batch_labels.append(label)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot total time taken for method e
axs[0].plot(batch_labels, total_times, marker='o', linestyle='None', color='b')
axs[0].set_title('Total Time Taken for Method E')
axs[0].set_ylabel('Total Time (seconds)')
axs[0].tick_params(axis='x', rotation=90)

# Plot read loop times
axs[1].plot(batch_labels, avg_read_time_0_45, marker='o', linestyle='None', label='Avg Read Time (0-45)', color='g')
axs[1].plot(batch_labels, max_read_time_0_45, marker='o', linestyle='None', label='Max Read Time (0-45)', color='r')
axs[1].plot(batch_labels, min_read_time_0_45, marker='o', linestyle='None', label='Min Read Time (0-45)', color='b')

axs[1].set_title('Read Loop Times')
axs[1].set_ylabel('Time (seconds)')
axs[1].set_xlabel('Batch')

axs[2].plot(batch_labels, avg_read_time_46_98, marker='x', linestyle='None', label='Avg Read Time (46-98)', color='y')
axs[2].plot(batch_labels, max_read_time_46_98, marker='x', linestyle='None', label='Max Read Time (46-98)', color='m')
axs[2].plot(batch_labels, min_read_time_46_98, marker='x', linestyle='None', label='Min Read Time (46-98)', color='c')

axs[2].set_title('Read Loop Times')
axs[2].set_ylabel('Time (seconds)')
axs[2].set_xlabel('Batch')
axs[2].legend()
axs[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
plt.savefig('/Users/rcosenti/RTB/RTB-RFP/SuperDark_Stress_Tests/Batch_devRFP_Testing.png')
