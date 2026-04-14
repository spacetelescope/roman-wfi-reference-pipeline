import numpy as np
import pandas as pd
import glob
import asdf
from astropy.io import fits
import os
import math
from enum import Enum
import time

import roman_datamodels as rdm

from multiprocessing import Pool
import logging
from datetime import datetime

from romancal.dq_init import DQInitStep
from romancal.refpix import RefPixStep

from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline

import warnings
from asdf.exceptions import AsdfPackageVersionWarning, AsdfConversionWarning


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

warnings.filterwarnings(
    "ignore",
    category=AsdfConversionWarning,
    module="asdf"
)

os.environ["ROMAN_VALIDATE"] = "false"
asdf.get_config().validate_on_read = False

log_file = f"fgs_mask_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Thresholds and constants
E_PER_DN = 2

DEAD_SIGMA_THR = 5.0
HOT_THR = 5.0 / E_PER_DN
SUPERHOT_THR = 20.0 # DN
HIGH_CDS_THR = 22 / E_PER_DN
LOW_QE_THR = 0.3
BAD_FLAT_THR = 0.0

NPROCESSES = 4

# Current FGS BPM flags
class flags(np.uint32, Enum):
    GOOD = 0
    GW_AFFECTED_DATA = 2**4
    PERSISTENCE = 2**5
    DEAD = 2**10
    HOT_PIXEL = 2**11
    FLAT_FIELD = 2**18
    HIGH_CDS_NOISE = 2**26
    LOW_QE_OPTICAL = 2**27
    REFERENCE_PIXEL = 2**31
    OTHER_BAD_PIXEL = 2**30
    TFPN_NEG = 2**0
    TFPN_POS = 2**7
    HOT_FROM_GW = 2**29


class DarkDataCube():
    """
    Datacube used to create the CDS noise datacube + dark rate image

    --- COPIED FROM RFP READNOISE MODULE ---
    ReadNoiseDataCube class derived from DataCube.
    Handles ReadNoise specific cube calculations
    Provide common fitting methods to calculate cube properties, such as rate and intercept images, for reference types.

    Parameters
    -------
    self.ref_type_data: input data array in cube shape
    self.wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
    """

    def __init__(self, ref_type_data):
        self.data=ref_type_data
        self.rate_image = None  # The linear slope coefficient of the fitted data cube.
        self.rate_image_err = None  # uncertainty in rate image
        self.intercept_image = None
        self.num_reads, self.num_i_pixels, self.num_j_pixels = self.data.shape
        self.time_array = np.arange(self.num_reads) * 3.16247
        self.intercept_image_err = (
            None  # uncertainty in intercept image (could be variance?)
        )
        self.ramp_model = None  # Ramp model of data cube.
        self.coeffs_array = None  # Fitted coefficients to data cube.
        self.covars_array = None  # Fitted covariance array to data cube.

    def fit_cube(self, degree=1):
        """
        fit_cube will perform a linear least squares regression using np.polyfit of a certain
        pre-determined degree order polynomial. This method needs to be intentionally called to
        allow for pipeline inputs to easily be modified.

        Parameters
        -------
        degree: int, default=1
            Input order of polynomial to fit data cube. Degree = 1 is linear. Degree = 2 is quadratic.
        """
        # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.
        try:
            self.coeffs_array, self.covars_array = np.polyfit(
                self.time_array,
                self.data.reshape(len(self.time_array), -1),
                degree,
                full=False,
                cov=True,
            )
            # Reshape the parameter slope array into a 2D rate image.
            #TODO the reshape and indices here are for linear degree fit = 1 only; update to handle quadratic al
            self.rate_image = self.coeffs_array[0].reshape(
                self.num_i_pixels, self.num_j_pixels
            )
            # Reshape the parameter y-intercept array into a 2D image.
            self.intercept_image = self.coeffs_array[1].reshape(
                self.num_i_pixels, self.num_j_pixels
            )
        except (TypeError, ValueError) as e:
            logging.info(e)

    def make_ramp_model(self, order=1):
        """
        make_data_cube_model uses the calculated fitted coefficients from fit_cube() to create
        a linear (order=1) or quadratic (order=2) model to the input data cube.

        NOTE: The default behavior for fit_cube() and make_model() utilizes a linear fit to the input
        data cube of which a linear ramp model is created.

        Parameters
        -------
        order: int, default=1
            Order of model to the data cube. Degree = 1 is linear. Degree = 2 is quadratic.
        """
        # Reshape the 2D array into a 1D array for input into np.polyfit().
        # The model fit parameters p and covariance matrix v are returned.
        try:
            # Reshape the returned covariance matrix slope fit error.
            # rate_var = v[0, 0, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO -VERIFY USE
            # returned covariance matrix intercept error.
            # intercept_var = v[1, 1, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO - VERIFY USE
            self.ramp_model = np.zeros((self.num_reads, self.num_i_pixels, self.num_j_pixels), dtype=np.float32)
            if order == 1:
                # y = m * x + b
                # where y is the pixel value for every read,
                # m is the slope at that pixel or the rate image,
                # x is time (this is the same value for every pixel in a read)
                # b is the intercept value or intercept image.
                for tt in range(0, len(self.time_array)):
                    self.ramp_model[tt, :, :] = (
                        self.rate_image * self.time_array[tt]
                        + self.intercept_image
                    )
            elif order == 2:
                # y = ax^2 + bx + c
                # where we dont have a single rate image anymore, we have coefficients
                for tt in range(0, len(self.time_array)):
                    a, b, c = self.coeffs_array
                    self.ramp_model[tt, :, :] = (
                        a * self.time_array[tt] ** 2
                        + b * self.time_array[tt]
                        + c
                    )
            else:
                raise ValueError(
                    "This function only supports polynomials of order 1 or 2."
                )
        except (ValueError, TypeError) as e:
            logging.info(e)


class FlatDataCube:
    """
    Data cube to create the slope map
    --- COPIED FROM RFP MASK MODULE ---
    Lightweight polynomial fitter for ramp cubes. From Bernie R. and Sarah Betti.
    """
    def __init__(self, nz: int, degree: int = 1):
        self.nz = nz
        self.degree = degree
        self.z = np.arange(nz)

        # Precompute basis matrix and its pseudo-inverse
        self.B = np.vander(self.z, N=degree + 1, increasing=True)
        self.pinvB = np.linalg.pinv(self.B)
        self.B_x_pinvB = self.B @ self.pinvB

    def fit(self, data):
        """
        Fits the polynomial to the data.
        Returns the coefficients of the fit.
        """
        return (self.pinvB @ data.reshape(self.nz, -1)).reshape((-1, *data.shape[1:]))

    def model(self, data):
        """
        Models the data based on the polynomial fit.
        Returns the modeled data.
        """
        return (self.B_x_pinvB @ data.reshape(self.nz, -1)).reshape((-1, *data.shape[1:]))


def change_coord_to_det(arr, det):
    '''
    Change the detector coordinates from DETECTOR to SCIENCE (run again to undo). Dependent on detector.
    Code from Sarah Betti
    '''
    # Detector coordinate positions; GSFC uses detector, SOC uses science
    detector_pos = {
        "WFI01": "upper left",
        "WFI02": "upper left",
        "WFI03": "lower right",
        "WFI04": "upper left",
        "WFI05": "upper left",
        "WFI06": "lower right",
        "WFI07": "upper left",
        "WFI08": "upper left",
        "WFI09": "lower right",
        "WFI10": "upper left",
        "WFI11": "upper left",
        "WFI12": "lower right",
        "WFI13": "upper left",
        "WFI14": "upper left",
        "WFI15": "lower right",
        "WFI16": "upper left",
        "WFI17": "upper left",
        "WFI18": "lower right"
    }

    position = detector_pos[det]

    if position == "lower right":
        return arr[:, ::-1]

    else:
        return arr[::-1]


def _get_slope(file):
    """
    Extracts the slope (linear term) of the data using polynomial fitting.
    """
    with asdf.open(file, memmap=True) as rf:

        data = rf["roman"]["data"]
        data = data.value if hasattr(data, "value") else data
        datacube = FlatDataCube(data.shape[0], degree=1)

        # Extract the linear coefficient
        slope = datacube.fit(data)[1]

    return slope


def create_super_slope_image(filelist, multip):
    """
    Fit a slope to each file in filelist, then average
    all slopes together to create a super slope image.
    """
    # Speed up slope calculation with Pool's map function
    if multip:
        with Pool(processes=NPROCESSES) as pool:
            slopes = pool.map(_get_slope, filelist)

    else:
        slopes = [_get_slope(file) for file in filelist]

    super_slope_image = np.nanmean(slopes,
                                   axis=0)

    return super_slope_image


def create_normalized_image(filelist, multip):
    """Create super slope image from filelist, then normalize."""
    super_slope = create_super_slope_image(filelist, multip)

    return super_slope / np.nanmean(super_slope)


def compute_cds_noise_from_datacube(rn_cube, cds_noise_path):
    """Compute CDS noise image. COPIED FROM READNOISE MODULE"""

    read_diff_cube = np.zeros((math.ceil(rn_cube.num_reads / 2), rn_cube.num_i_pixels, rn_cube.num_j_pixels), dtype=np.float32)

    for i_read in range(0, rn_cube.num_reads - 1, 2):
        # Avoid index error if num_reads is odd and disregard the last read because it does not form a pair.
        rd1 = (rn_cube.ramp_model[i_read, :, :] - rn_cube.data[i_read, :, :])
        rd2 = (rn_cube.ramp_model[i_read + 1, :, :] - rn_cube.data[i_read + 1, :, :])

        read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1

        rn_cube.cds_noise = np.std(read_diff_cube, axis=0)

    fits.writeto(cds_noise_path,
                 data=rn_cube.cds_noise,
                 overwrite=True)


def run_romancal(file, outpath):
    """ Function to run DQInit and RefPix steps on raw data """
    with rdm.open(file) as f:

        dq_data = DQInitStep.call(f)
        _ = RefPixStep.call(dq_data, save_results=True, output_dir=outpath)

    return


def run_workflow(overwrite=False):
    # Modify line below to run on all detectors
    for i in np.arange(1, 19):
        det = f"WFI{i:02d}"

        logging.info(f"Running FGS mask workflow on {det}")

        basedir = f"/path/to/out/for/fgs-mask/{det}"
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # List of raw darks and flats
        shortdarks = glob.glob(f"/roman/path/to/raw/OTP00639_TotalNoiseNoEWA_TV2a_R1_MCEB/**/*{det}*asdf")
        flats = glob.glob(f"/roman/path/to/raw/OTP00615_SmoothDarkptA_TV2a_R1_MCEB/**/*{det}*asdf")

        # Getting the number of exposures expected for CAR-094
        ndarks, nflats = 10, 30
        shortdarks = shortdarks[:ndarks] if len(shortdarks) > ndarks else shortdarks
        flats = flats[:nflats] if len(flats) > nflats else flats

        logging.info("Short darks used: ", shortdarks)
        logging.info("Flats used: ", flats)

        # Performing IRRC correction
        try:
            outpath_flats = os.path.join(basedir, "irrc_corr/flats")
            outpath_darks = os.path.join(basedir, "irrc_corr/darks")

            # Create output directory for files run through romancal
            if not os.path.exists(outpath_darks):
                os.makedirs(outpath_flats,
                            exist_ok=True)

                os.makedirs(outpath_darks,
                            exist_ok=True)

            # Check if the files have already been IRRC corrected
            files_exist = (
                len(os.listdir(outpath_darks)) == ndarks
                and len(os.listdir(outpath_flats)) == len(flats)
            )

            # If overwrite or the files don't exist, need to run romancal
            if overwrite or not files_exist:
                args = [(file, outpath_flats) for file in flats] + [(file, outpath_darks) for file in shortdarks]

                with Pool(processes=NPROCESSES) as pool:
                    _ = pool.starmap(run_romancal, args)

        except Exception as e:
            logging.info(f"Error processing files: {e}")

        finally:
            if 'pool' in locals():
                pool.close()
                pool.join()

        logging.info("Finished running IRRC on all raw data used")

        # Creating superdark from the short darks using the RFP superdark code
        irrc_shortdarks = glob.glob(os.path.join(outpath_darks, f"*{det}*"))
        superdark_path = os.path.join(basedir, "superdark.asdf")

        nreads = 55

        # Create the superdark if it doesn't already exist
        if overwrite or not os.path.exists(superdark_path):
            logging.info("Beginning superdark creation")

            dark_pipe = DarkPipeline(det)
            dark_pipe.prep_superdark_file(full_file_list=irrc_shortdarks,
                                          outfile=superdark_path,
                                          full_file_num_reads=nreads)

            logging.info("Finished generating superdark")

        logging.info("Loading superdark")
        with asdf.open(superdark_path, memmap=True) as af:
            data = af["roman"]["data"]
            superdark = data.value if hasattr(data, "value") else data
            superdark = np.asarray(superdark)

        # Creating ReadNoise object from superdark (for CDS noise + rate image)
        cds_noise_path = os.path.join(basedir, "cds_noise.fits")
        darkrate_image_path = os.path.join(basedir, "darkrate.fits")

        logging.info("Computing CDS noise and fitting ramp to produce rate image")

        # Generate data cube object
        rn_cube = DarkDataCube(superdark)

        # Prep CDS noise computations
        rn_cube.fit_cube(degree=1)
        rn_cube.make_ramp_model(order=1)

        # Compute and write CDS noise image
        compute_cds_noise_from_datacube(rn_cube, cds_noise_path)

        # Write the dark rate image
        fits.writeto(darkrate_image_path,
                     data=rn_cube.rate_image,
                     overwrite=True)

        logging.info("CDS Noise and rate images created.")

        # Creating normalized super slope image and super slope image
        irrc_flats = glob.glob(os.path.join(outpath_flats, f"*{det}*"))

        logging.info("Creating normalized slope image and super slope image")

        # Creating superslope and norm superslope images
        super_slope_image = create_super_slope_image(irrc_flats, multip=True)
        normalized_image = create_normalized_image(irrc_flats, multip=True)

        super_slope_path = os.path.join(basedir, "super_slope.fits")
        normalized_image_path = os.path.join(basedir, "normalized_image.fits")

        # Saving the super slope and normalized images
        fits.writeto(super_slope_path,
                     data=super_slope_image,
                     overwrite=True)

        fits.writeto(normalized_image_path,
                     data=normalized_image,
                     overwrite=True)

        logging.info("Finished creating normalized slope image and super slope image")

        # Using thresholds to identify bad pixels
        # Empty mask of zeros to be filled in with bitvals
        mask = np.zeros((4096, 4096), dtype=np.uint32)

        logging.info("Beginning bad pixel identification")

        # Identifying DEAD pixels
        median_slope = np.median(super_slope_image)
        std_slope = np.std(super_slope_image)

        dead_threshold = median_slope - (DEAD_SIGMA_THR * std_slope)
        dead_mask = (super_slope_image < dead_threshold)
        mask[dead_mask] += flags.DEAD

        # Identifying HOT and SUPERHOT pixels
        hot_mask = (rn_cube.rate_image > HOT_THR)
        mask[hot_mask] += flags.HOT_PIXEL

        superhot_mask = (rn_cube.rate_image > SUPERHOT_THR)
        mask[superhot_mask] += flags.HOT_PIXEL

        # Identifying HIGH_CDS_NOISE
        cds_mask = (rn_cube.cds_noise > HIGH_CDS_THR)
        mask[cds_mask] += flags.HIGH_CDS_NOISE

        # Identifying LOW_QE pixels
        qe_mask = (normalized_image < LOW_QE_THR)
        mask[qe_mask] += flags.LOW_QE_OPTICAL

        # Identifying BAD_FLAT_FIELD pixels
        flat_mask = (normalized_image < BAD_FLAT_THR)
        mask[flat_mask] += flags.FLAT_FIELD

        logging.info("Writing mask to output directory")

        # Writing mask to disk
        mask_path = os.path.join(basedir, f"mask_{det}.fits")
        fits.writeto(mask_path,
                     data=mask,
                     overwrite=True)

        # PSS expects the mask as FITS boolean file in DETECTOR coordinates (not SCIENCE)
        logging.info("Transforming mask to boolean mask in DETECTOR coordinates")
        binary_mask = (mask != 0).astype("uint8")
        binary_mask_det = change_coord_to_det(binary_mask, det)

        logging.info("Writing transformed boolean mask to FITS")
        binary_mask_path = os.path.join(basedir, f"binary_mask_{det}.fits")
        fits.writeto(binary_mask_path,
                     data=binary_mask_det,
                     overwrite=True)

        logging.info("Finished running FGS mask workflow on ", det)


if __name__ == "__main__":

    start_time = time.time()

    # The main workflow to run thorugh all detectors
    run_workflow()

    end_time = time.time()

    elapsed = end_time - start_time

    # Print time elapsed
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    logging.info(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")
