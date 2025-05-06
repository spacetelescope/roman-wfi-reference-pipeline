import argparse
from pathlib import Path

import asdf
import numpy as np
from astropy.time import Time

from wfi_reference_pipeline.constants import REF_TYPE_DARK, WFI_FRAME_TIME, WFI_MODE_WIM
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.utilities.data_functions import get_science_pixels_cube
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

# TODO edit below if needed and add extra meta for testing/validation
# Define your metadata, including only the 'exposure' information
meta = {
    'exposure': {
        'id': 1,
        'type': 'WFI_DARK',
        'start_time': Time('2024-01-01T00:00:00', scale='utc'),
        'mid_time': Time('2021-01-01T00:01:26.660', scale='utc'),
        'end_time': Time('2021-01-01T00:02:53.320', scale='utc'),
        'start_time_mjd': 59215.0,
        'mid_time_mjd': 59215.00100300926,
        'end_time_mjd': 59215.00200601852,
        'start_time_tdb': 59215.00080073967,
        'mid_time_tdb': 59215.00180374893,
        'end_time_tdb': 59215.00280675819,
        'ngroups': 10,  # TODO
        'nframes': 10,  # TODO
        'data_problem': False,
        'sca_number': 1,
        'gain_factor': 2,
        'integration_time': 10.0,  # TODO
        'elapsed_exposure_time': 10.0,  # TODO
        'frame_divisor': 1,  # TODO
        'groupgap': 0,
        'frame_time': 1.0,  # TODO
        'group_time': 1.0,  # TODO
        'exposure_time': 10.0,  # TODO
        'effective_exposure_time': 10.0,  # TODO
        'duration': 10.0,  # TODO
        'ma_table_name': 'VALIDATION DARK MA TABLE',  # TODO
        'ma_table_number': 999,  # TODO
        'level0_compressed': True,
        'read_pattern': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],  # TODO
        'truncated': False
    }
}

def generate_short_dark_files(n_files=9, n_reads=10, output_dir='/grp/roman/RFP/DEV/scratch'):
    """
    Generate short dark files with controlled inputs using simulate_dark_reads and save as ASDF files.

    Parameters
    ----------
    output_dir: str
        Directory where the files will be saved.
    n_files: int, default = 9
        Number of short dark files to generate.
    n_reads: int, default = 10
        Number of reads for each file.

    The function generates files that are described by:
        (1) short dark file with a dark rate of 1
        (5) short dark files with a dark rate of 2
        (1) short dark file with a dark rate of 3
        (1) short dark file with a dark rate of 8
        (1) short dark file with a dark rate of 25

    """
    rate_values = [1, 2, 2, 2, 2, 2, 3, 8, 25]  # Values for each file

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    short_files=[]
    for i in range(n_files):
        read_cube, rate_image = simulate_dark_reads(
            n_reads=n_reads,
            exp_time=WFI_FRAME_TIME[WFI_MODE_WIM],
            dark_rate=rate_values[i],  # Set the dark rate to the desired value (1, 2, 3, 5, 10)
            dark_rate_var=0.0,  # No variance to maintain uniform values
            num_hot_pix=0,
            num_hot_pix_var=0,
            num_warm_pix=0,
            num_warm_pix_var=0,
            num_dead_pix_var=0,
            noise_mean=0.0,
            noise_var=0.0
        )

        # Create the ASDF file
        file_name = f'short_dark{i + 1}_WFI01.asdf'
        file_path = output_path / file_name

        # Create an ASDF file structure
        tree = {
            'asdf_library': {
                'author': 'The ASDF Developers',
                'homepage': 'http://github.com/asdf-format/asdf',
                'name': 'asdf',
                'version': '3.0.1'
            },
            'roman': {
                'meta': meta,
                'data': read_cube
            }
        }

        # Write to ASDF
        with asdf.AsdfFile(tree) as af:
            af.write_to(file_path)
            short_files.append(file_path)

    return short_files


def generate_long_dark_files(n_files=2, n_reads=20, output_dir='/grp/roman/RFP/DEV/scratch'):
    """
    Generate long dark files with controlled inputs using simulate_dark_reads and save as ASDF files.

    Parameters
    ----------
    output_dir : str
        Directory where the files will be saved.
    n_files : int, optional
        Number of short dark files to generate (default is 5).
    n_reads : int, optional
        Number of reads for each file (default is 10).

    The function generates files that are described by:
        (2) long dark file with a dark rate of 2
    """
    rate_values = [2, 2]  # Values for each file

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    long_files = []
    for i in range(n_files):
        read_cube, rate_image = simulate_dark_reads(
            n_reads=n_reads,
            exp_time=WFI_FRAME_TIME[WFI_MODE_WIM],
            dark_rate=rate_values[i],  # Set the dark rate to the desired value (2, 2)
            dark_rate_var=0.0,  # No variance to maintain uniform values
            num_hot_pix=0,
            num_hot_pix_var=0,
            num_warm_pix=0,
            num_warm_pix_var=0,
            num_dead_pix_var=0,
            noise_mean=0.0,
            noise_var=0.0
        )

        # Create the ASDF file
        file_name = f'long_dark{i + 1}_WFI01.asdf'
        file_path = output_path / file_name

        # Create an ASDF file structure
        tree = {
            'asdf_library': {
                'author': 'The ASDF Developers',
                'homepage': 'http://github.com/asdf-format/asdf',
                'name': 'asdf',
                'version': '3.0.1'
            },
            'roman': {
                'meta': meta,
                'data': read_cube
            }
        }

        # Write to ASDF
        with asdf.AsdfFile(tree) as af:
            af.write_to(file_path)
            long_files.append(file_path)

    return long_files

# Validate SuperDark function or class Rick pseudo code
def test_validate_superdark_sigma_clip_short_only_values_pass(short_files):
    test_meta = MakeTestMeta(REF_TYPE_DARK)

    # Set sigma clipping level to 1 (keep them the same when doing this test)
    sigma_clip_low_bound = 1
    sigma_clip_high_bound = 1

    # Generate superdark from only short darks
    dark_pipeline = DarkPipeline("WFI01")
    dark_pipeline.prep_superdark_file(short_file_list=short_files, short_dark_num_reads=10, long_dark_num_reads=0, sig_clip_sd_low=sigma_clip_low_bound, sig_clip_sd_high=sigma_clip_high_bound, outfile="validate_superdark_test_prepped_superdark_short.asdf")
    # Check 1-sigma rejection and dark rates.
    # Use the Dark() to compute the mean dark rate from the generated superdark.asdf file.
    dark = Dark(meta_data=test_meta.meta_dark, file_list=[dark_pipeline.superdark_file], outfile="validate_superdark_test_dark.asdf")
    # 1-sigma: (clips all values but the rate values associated with 2's)
    science_pixels_cube = get_science_pixels_cube(dark.data_cube.data) # only verify numbers with science pixels
    data_cube_array = np.array(science_pixels_cube)
    std_dev_all = np.nanstd(data_cube_array)
    mean_all = np.nanmean(data_cube_array)

    assert(np.isclose(std_dev_all, 17.46, rtol=1e-2, atol=1e-2))
    assert(np.isclose(mean_all, 33.44, rtol=1e-2, atol=1e-2))

    dark.make_rate_image_from_data_cube()
    average = np.mean(dark.dark_rate_image)
    assert(np.isclose(average, 2, rtol=1e-1, atol=1e-1))
    print(f"average {average}")
    #Reference pixels are set to zero, so just grabbing science pixels.
    print("PASSED - test_validate_superdark_sigma_clip_short_only_values_pass")


def test_validate_superdark_sigma_clip_values_pass(short_files, long_files):
    test_meta = MakeTestMeta(REF_TYPE_DARK)

    # Set sigma clipping level 3 sigma (keep them the same when doing this test)
    sigma_clip_low_bound = 3
    sigma_clip_high_bound = 3

    # Generate superdark from only short darks
    dark_pipeline = DarkPipeline("WFI01")
    dark_pipeline.prep_superdark_file(short_file_list=short_files, long_file_list=long_files, short_dark_num_reads=10, long_dark_num_reads=20, sig_clip_sd_low=sigma_clip_low_bound, sig_clip_sd_high=sigma_clip_high_bound, outfile="validate_superdark_test_prepped_superdark.asdf")

    # Generate dark ref_type to get data
    dark = Dark(meta_data=test_meta.meta_dark, file_list=[dark_pipeline.superdark_file], outfile="validate_superdark_test_dark.asdf")

    # 3-sigma:
    # Each value represents all values in a slice in the data cube
    science_pixels_cube = get_science_pixels_cube(dark.data_cube.data) # only verify numbers with science pixels
    data_cube_array = np.array(science_pixels_cube)
    std_dev_all = np.nanstd(data_cube_array)
    mean_all = np.nanmean(data_cube_array)

    assert(np.isclose(std_dev_all, 32.47, rtol=1e-2, atol=1e-2))
    assert(np.isclose(mean_all, 68.85, rtol=1e-2, atol=1e-2))

    dark.make_rate_image_from_data_cube()
    average = np.mean(dark.dark_rate_image)
    assert(np.isclose(average, 2, rtol=1e-1, atol=1e-1))
    print("PASSED - test_validate_superdark_sigma_clip_values_pass")

# Check if files exist for validation test, if not make them with above, if they do proceed
def generate_files(rebuild_files, input_dir):

    input_path = Path(input_dir)

    files = [file for file in input_path.iterdir() if file.is_file()]
    asdf_files = [file for file in files if ".asdf" in file.name]
    short_files = [file for file in asdf_files if "short" in file.name]
    long_files = [file for file in asdf_files if "long" in file.name]

    if rebuild_files:
        print("Removing pre-existing dark files for test")
        for file in short_files:
            file.unlink()
            print(f"Deleted: {file}")
        short_files = []

        for file in long_files:
            file.unlink()
            print(f"Deleted: {file}")
        long_files=[]

    if len(short_files) == 0:
        short_files = generate_short_dark_files()
    if len(long_files) == 0:
        long_files = generate_long_dark_files()

    return short_files, long_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate that a superdark is constructed properly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-r", "--rebuild", action="store_true", help="Rebuild the dark files")
    parser.add_argument("-p", "--path", type=str, help="Dark Files Path.", default="/grp/roman/RFP/DEV/scratch"
    )
    args = parser.parse_args()

    # Make the files (or dont if they already exist)
    if args.rebuild:
        rebuild = True
        print("-r: Rebuilding Dark Files")
    else:
        rebuild = False

    print(f"Generated Dark File Path: {args.path}")
    short_files, long_files = generate_files(rebuild, args.path)

    test_validate_superdark_sigma_clip_values_pass(short_files, long_files)
    test_validate_superdark_sigma_clip_short_only_values_pass(short_files)
