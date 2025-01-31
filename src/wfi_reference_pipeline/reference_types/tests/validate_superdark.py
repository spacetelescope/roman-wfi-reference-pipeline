from pathlib import Path
from wfi_reference_pipeline.utilities.data_functions import get_science_pixels_cube
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
import asdf
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.constants import REF_TYPE_DARK
from astropy.time import Time
import numpy as np

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

test_meta = MakeTestMeta(REF_TYPE_DARK)
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

        read_cube = np.full((n_reads, 4096, 4096), rate_values[i])

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

        read_cube = np.full((n_reads, 4096, 4096), rate_values[i])

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
def test_validate_superdark_sigma_clip_short_only_values_pass(input_dir='/grp/roman/RFP/DEV/scratch'):
    # Make the files (or dont if they already exist)
    short_files, long_files = generate_files(input_dir)

    # Set sigma clipping level to 1 (keep them the same when doing this test)
    sigma_clip_low_bound = 1
    sigma_clip_high_bound = 1

    # Generate superdark from only short darks
    dark_pipeline = DarkPipeline()
    dark_pipeline.prep_superdark_file(short_file_list=short_files, short_dark_num_reads=10, long_dark_num_reads=0, sig_clip_sd_low=sigma_clip_low_bound, sig_clip_sd_high=sigma_clip_high_bound, outfile="validate_superdark_test_prepped_superdark_short.asdf")
    # Check 1-sigma rejection and dark rates.
    # Use the Dark() to compute the mean dark rate from the generated superdark.asdf file.
    dark = Dark(meta_data=test_meta.meta_dark, file_list=[dark_pipeline.superdark_file], outfile="validate_superdark_test_dark.asdf")
    # 1-sigma: (clips all values but 2)
    # [nan  2.  2.  2.  2.  2. nan nan nan]
    # mean: 2.0
    # std_dev: 0.0
    science_pixels_cube = get_science_pixels_cube(dark.data_cube.data) # only verify numbers with science pixels
    data_cube_array = np.array(science_pixels_cube)
    std_dev_all = np.nanstd(data_cube_array)
    mean_all = np.nanmean(data_cube_array)

    assert(std_dev_all == 0)
    assert(mean_all == 2)
    print("PASSED - test_validate_superdark_sigma_clip_short_only_values_pass")


def test_validate_superdark_sigma_clip_values_pass(input_dir='/grp/roman/RFP/DEV/scratch'):
    # Make the files (or dont if they already exist)
    short_files, long_files = generate_files(input_dir)

    # Set sigma clipping level 3 sigma (keep them the same when doing this test)
    sigma_clip_low_bound = 3
    sigma_clip_high_bound = 3

    # Generate superdark from only short darks
    dark_pipeline = DarkPipeline()
    dark_pipeline.prep_superdark_file(short_file_list=short_files, long_file_list=long_files, short_dark_num_reads=10, long_dark_num_reads=20, sig_clip_sd_low=sigma_clip_low_bound, sig_clip_sd_high=sigma_clip_high_bound, outfile="validate_superdark_test_prepped_superdark.asdf")

    # Generate dark ref_type to get data
    dark = Dark(meta_data=test_meta.meta_dark, file_list=[dark_pipeline.superdark_file], outfile="validate_superdark_test_dark.asdf")

    # 3-sigma:
    # [ 1.  2.  2.  2.  2.  2.  3.  8. nan  2.  2.]
    # Unique values are ten slices of 2.6 (short and long average) and ten slices of 2 (long only)
    # mean: around 2.3
    # std_dev: around 0.3
    science_pixels_cube = get_science_pixels_cube(dark.data_cube.data) # only verify numbers with science pixels
    data_cube_array = np.array(science_pixels_cube)
    std_dev_all = np.nanstd(data_cube_array)
    mean_all = np.nanmean(data_cube_array)

    assert(np.isclose(std_dev_all, 0.3, rtol=1e-2, atol=1e-2))
    assert(np.isclose(mean_all, 2.3, rtol=1e-2, atol=1e-2))
    print("PASSED - test_validate_superdark_sigma_clip_values_pass")


    dark.make_rate_image_from_data_cube()
    dark.make_ma_table_resampled_data(num_resultants=8, num_reads_per_resultant=6)
    dark.update_data_quality_array()




    # Use the Dark() to compute the mean dark rate from the generated
    # superdark.asdf file.

    # NOTE checking the dark rate might be easier than inspecting values of large cubes and more comprehensive in utilizing
    # the RFP Dark() module and fitting.

    # Next step: more scientifically interesting validation would be to include noise and variance of the rates and assert
    # that the measured dark rate from Dark() with the superdark as input produces a dark rate image within some tolerance.






# Check if files exist for validation test, if not make them with above, if they do proceed
def generate_files(input_dir='/grp/roman/RFP/DEV/scratch'):

    input_path = Path(input_dir)
    files = [file for file in input_path.iterdir() if file.is_file()]
    asdf_files = [file for file in files if ".asdf" in file.name]
    short_files = [file for file in asdf_files if "short" in file.name]
    long_files = [file for file in asdf_files if "long" in file.name]

    if len(short_files) == 0:
        short_files = generate_short_dark_files()
    if len(long_files) == 0:
        long_files = generate_long_dark_files()

    return short_files, long_files


if __name__ == "__main__":
    #test_validate_superdark_sigma_clip_short_only_values_pass()
    test_validate_superdark_sigma_clip_values_pass()
