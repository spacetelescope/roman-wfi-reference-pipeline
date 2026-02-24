"""
This is to test the timing of the full Mask() module workflow
when supplied with both flats and darks.
"""
import glob
import os
import re
import warnings

import numpy as np
from asdf.exceptions import AsdfPackageVersionWarning
from astropy.io import fits

from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# If True, re-run the telegraph and RC identification step on decreasing numbers of jumpy pixels.
# The number of jumpy pixels will decrease by a factor of 10 until < 100
PERFORM_NUM_JUMPY_PIXELS_CHECK = True

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="asdf"
)

warnings.filterwarnings(
    "ignore",
    category=AsdfPackageVersionWarning,
    module="asdf"
)

os.environ["ROMAN_VALIDATE"] = "false"

def test_full_mask(outpath, flats):
    """
    Perform the full Mask() workflow on long darks and flats.
    This is how a user would call the Mask() module, with no
    modifications to the algorithm parameters.
    """
    tmp = MakeDevMeta(ref_type="MASK")

    tmp.meta_mask.use_after = '2020-05-01T00:00:00.000'
    tmp.meta_mask.author = "A user"
    tmp.meta_mask.instrument_detector = DET
    tmp.meta_mask.description = "Testing new mask from dark modules. "

    rfp_mask = Mask(meta_data=tmp.meta_mask,
                    file_list=longdarks,
                    outfile=os.path.join(outpath, f"mask_test_{DET}.asdf"),
                    clobber=True)

    rfp_mask.make_mask_image(from_smoothed=False,
                             flat_filelist=flats,
                             intermediate_path=outpath)

    rfp_mask.generate_outfile()


def create_jump_products_reduced_num(file_jump, outpath, divideby=10):
    """
    This function is used for timing the Mask() module
    function set_rc_tel_pixels(), which is the most time-intensive
    step in the Mask() module.

    Returns a list of the filepaths to the new jump_products.fits
    """
    filepaths = []

    with fits.open(file_jump) as hdu:

        jump_count = hdu[1].data # (4096, 4096)
        jump_mask = hdu[2].data # (349, 4096, 4096)

        jump_indices = np.argwhere(jump_count)
        n_jumpy_pix = len(jump_indices)

        # Creating list of varying n_jumpy_pix to test mem usage/timing
        n_iter = []
        n = n_jumpy_pix / 2

        while n >= 100:
            n_iter.append(int(n))
            n /= divideby

        rng = np.random.default_rng()

        # Creating jump_products.fits file with N jumpy pixels
        for n in n_iter:

            # Modifying jump_count arr to only have N pixels with jumps != 0
            n_remove = n_jumpy_pix - n

            # Randomly select indicies in jump_indices
            chosen_pix = rng.choice(n_jumpy_pix, size=n_remove, replace=False)

            # Coordinates for pixels that will have their jumps "masked"
            flip_coords = jump_indices[chosen_pix]

            rows = flip_coords[:, 0]
            cols = flip_coords[:, 1]

            # Masking the jump_count array
            jump_count_reduced = jump_count.copy()
            jump_count_reduced[rows, cols] = 0

            # Masking the jump_mask array
            jump_mask_reduced = jump_mask.copy()
            jump_mask_reduced[:, rows, cols] = False

            # Creating the new FITS file
            hdu_count = fits.ImageHDU(
                data=jump_count_reduced.astype(np.int16),
                name="JUMP_COUNT",
            )

            hdu_mask = fits.ImageHDU(
                data=jump_mask_reduced.astype(np.uint8),
                name="JUMP_MASK",
            )

            # Writing the reduced jump file
            hdul = fits.HDUList([
                fits.PrimaryHDU(),
                hdu_count,
                hdu_mask,
            ])

            outfile = os.path.join(outpath, f"jump_products_n{n}.fits")
            hdul.writeto(outfile, overwrite=True)

            filepaths.append(outfile)

    return filepaths


def extract_njumps_from_filepath(file):
    """
    Extract the number of pixels with jumps from the file string.
    """
    return re.search(r'jump_products_n(?P<njumps>\d+)', file).group("njumps")


def run_mask_on_subset_of_jumps(outpath, njumps, superdark_path, jump_path):
    """
    Run the update_mask_from_darks code with the superdark
    made from the test_full_mask() run.
    """

    tmp = MakeDevMeta(ref_type="MASK")

    tmp.meta_mask.use_after = '2020-05-01T00:00:00.000'
    tmp.meta_mask.author = "A user"
    tmp.meta_mask.instrument_detector = DET
    tmp.meta_mask.description = f"Timing set_rc_tel_pixels() on file with {njumps} pixels with jumps."

    rfp_mask = Mask(meta_data=tmp.meta_mask,
                    file_list=longdarks,
                    outfile=os.path.join(outpath, f"mask_test_{DET}_n{njumps}.asdf"),
                    clobber=True)

    rfp_mask.make_mask_image(intermediate_path=outpath,
                             superdark_path=superdark_path,
                             jump_path=jump_path)

    rfp_mask.generate_outfile()


if __name__ == "__main__":

    DET = "WFI09"

    longdarks = glob.glob(f"path/to/longdarks/*{DET}*.asdf")
    flats = glob.glob(f"path/to/flats/*{DET}*.asdf")

    outpath = "/path/to/output"

    test_full_mask(outpath)

    superdark_path = os.path.join(outpath, "superdark.asdf")

    if PERFORM_NUM_JUMPY_PIXELS_CHECK:

        print("Performing checking subsets of jumpy pixels")

        file_jump = os.path.join(outpath, "jump_products.fits")

        jump_filepaths = create_jump_products_reduced_num(file_jump, outpath)

        for file in jump_filepaths:

            print(f"Running RFP code on {file}")

            parent_dir = os.path.dirname(file)
            njumps = extract_njumps_from_filepath(file)

            run_mask_on_subset_of_jumps(parent_dir, njumps, superdark_path, file)
