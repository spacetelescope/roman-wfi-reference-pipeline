import glob
import logging
import os
from enum import Enum
from multiprocessing import Pool

import asdf
import numpy as np
from astropy.io import fits

from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.wfi_meta_fgs_mask import WFIMetaFGSMask

from ..reference_type import ReferenceType


# SAPP TODO - should these flags be imported from romandatamodels, else stored someplace else?
class Flags(np.uint32, Enum):
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


class FGSMask(ReferenceType):
    """
    Class FGSMask() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        outfile="roman_fgs_mask.asdf",
        clobber=False,
    ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            outfile=outfile,
            clobber=clobber,
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaFGSMask):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaFGSMask"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI fgs mask reference file."

    def make_fgs_mask_image(self):
        # Modify line below to run on all detectors

        # SAPP TODO FIGURE OUT WHAT OF THESE ELEMENTS ARE ALREADY IN THE PREP PIPELINE
        logging.info(f"Running FGS mask workflow on {det}")

        basedir = f"/path/to/out/for/fgs-mask/{det}"
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        # List of raw darks and flats
        shortdarks = glob.glob(
            f"/roman/path/to/raw/OTP00639_TotalNoiseNoEWA_TV2a_R1_MCEB/**/*{det}*asdf"
        )
        flats = glob.glob(
            f"/roman/path/to/raw/OTP00615_SmoothDarkptA_TV2a_R1_MCEB/**/*{det}*asdf"
        )

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
                os.makedirs(outpath_flats, exist_ok=True)

                os.makedirs(outpath_darks, exist_ok=True)

            # Check if the files have already been IRRC corrected
            files_exist = len(os.listdir(outpath_darks)) == ndarks and len(
                os.listdir(outpath_flats)
            ) == len(flats)

            # If overwrite or the files don't exist, need to run romancal
            if overwrite or not files_exist:
                args = [(file, outpath_flats) for file in flats] + [
                    (file, outpath_darks) for file in shortdarks
                ]

                with Pool(processes=NPROCESSES) as pool:
                    _ = pool.starmap(
                        run_romancal, args
                    )  # SAPP TODO - This should be in the prep pipeline

        except Exception as e:
            logging.info(f"Error processing files: {e}")

        finally:
            if "pool" in locals():
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
            dark_pipe.prep_superdark_file(
                full_file_list=irrc_shortdarks,
                outfile=superdark_path,
                full_file_num_reads=nreads,
            )

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
        rn_cube = Dark.DarkDataCube(
            superdark, self.meta_data.type
        )  # TODO - you mentioned getting this from readnoise and not dark? Lets import from where we need it, does this work for sierra's needs?

        # Prep CDS noise computations
        rn_cube.fit_cube(degree=1)
        rn_cube.make_ramp_model(order=1)

        # Compute and write CDS noise image
        compute_cds_noise_from_datacube(rn_cube, cds_noise_path)

        # Write the dark rate image
        fits.writeto(darkrate_image_path, data=rn_cube.rate_image, overwrite=True)

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
        fits.writeto(super_slope_path, data=super_slope_image, overwrite=True)

        fits.writeto(normalized_image_path, data=normalized_image, overwrite=True)

        logging.info("Finished creating normalized slope image and super slope image")

        # Using thresholds to identify bad pixels
        # Empty mask of zeros to be filled in with bitvals
        mask = np.zeros((4096, 4096), dtype=np.uint32)

        logging.info("Beginning bad pixel identification")

        # Identifying DEAD pixels
        median_slope = np.median(super_slope_image)
        std_slope = np.std(super_slope_image)

        # SAPP TODO - NOTE These constants are configurable in pipelines_config.yml and currently imported in fgs_mask_pipeline.py  Should this code be in prep?
        dead_threshold = median_slope - (DEAD_SIGMA_THR * std_slope)
        dead_mask = super_slope_image < dead_threshold
        mask[dead_mask] += Flags.DEAD

        # Identifying HOT and SUPERHOT pixels
        hot_mask = rn_cube.rate_image > HOT_THR
        mask[hot_mask] += Flags.HOT_PIXEL

        superhot_mask = rn_cube.rate_image > SUPERHOT_THR
        mask[superhot_mask] += Flags.HOT_PIXEL

        # Identifying HIGH_CDS_NOISE
        cds_mask = rn_cube.cds_noise > HIGH_CDS_THR
        mask[cds_mask] += Flags.HIGH_CDS_NOISE

        # Identifying LOW_QE pixels
        qe_mask = normalized_image < LOW_QE_THR
        mask[qe_mask] += Flags.LOW_QE_OPTICAL

        # Identifying BAD_FLAT_FIELD pixels
        flat_mask = normalized_image < BAD_FLAT_THR
        mask[flat_mask] += Flags.FLAT_FIELD

        logging.info("Writing mask to output directory")

        # Writing mask to disk
        mask_path = os.path.join(basedir, f"mask_{det}.fits")
        fits.writeto(mask_path, data=mask, overwrite=True)

        # PSS expects the mask as FITS boolean file in DETECTOR coordinates (not SCIENCE)
        logging.info("Transforming mask to boolean mask in DETECTOR coordinates")
        binary_mask = (mask != 0).astype("uint8")
        binary_mask_det = change_coord_to_det(binary_mask, det)

        logging.info("Writing transformed boolean mask to FITS")
        binary_mask_path = os.path.join(basedir, f"binary_mask_{det}.fits")
        fits.writeto(binary_mask_path, data=binary_mask_det, overwrite=True)

        logging.info("Finished running FGS mask workflow on ", det)

    def change_coord_to_det(self, arr, det):
        """
        Change the detector coordinates from DETECTOR to SCIENCE (run again to undo). Dependent on detector.
        Code from Sarah Betti
        """
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
            "WFI18": "lower right",
        }

        position = detector_pos[det]

        if position == "lower right":
            return arr[:, ::-1]

        else:
            return arr[::-1]

    def _get_slope(self, file):
        """
        Extracts the slope (linear term) of the data using polynomial fitting.
        """
        with asdf.open(file, memmap=True) as rf:
            data = rf["roman"]["data"]
            data = data.value if hasattr(data, "value") else data
            datacube = Flat.FlatDataCube(
                data.shape[0], degree=1
            )  # SAPP TODO, we want to use this but it takes different parameters

            # Extract the linear coefficient
            slope = datacube.fit(data)[1]

        return slope

    def create_super_slope_image(self, filelist, multip):
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

        super_slope_image = np.nanmean(slopes, axis=0)

        return super_slope_image

    def create_normalized_image(self, filelist, multip):
        """Create super slope image from filelist, then normalize."""
        super_slope = create_super_slope_image(filelist, multip)

        return super_slope / np.nanmean(super_slope)

    def compute_cds_noise_from_datacube(self, rn_cube, cds_noise_path):
        """Compute CDS noise image. COPIED FROM READNOISE MODULE"""  # TODO - I dont see where this was copied from?

        read_diff_cube = np.zeros(
            (
                math.ceil(rn_cube.num_reads / 2),
                rn_cube.num_i_pixels,
                rn_cube.num_j_pixels,
            ),
            dtype=np.float32,
        )

        for i_read in range(0, rn_cube.num_reads - 1, 2):
            # Avoid index error if num_reads is odd and disregard the last read because it does not form a pair.
            rd1 = rn_cube.ramp_model[i_read, :, :] - rn_cube.data[i_read, :, :]
            rd2 = rn_cube.ramp_model[i_read + 1, :, :] - rn_cube.data[i_read + 1, :, :]

            read_diff_cube[math.floor((i_read + 1) / 2), :, :] = rd2 - rd1

            rn_cube.cds_noise = np.std(read_diff_cube, axis=0)

        fits.writeto(cds_noise_path, data=rn_cube.cds_noise, overwrite=True)

    def calculate_error(self):
        """
        Abstract method not applicable to Gain.
        """

        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized by Gain().
        """

        pass

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """
        pass
