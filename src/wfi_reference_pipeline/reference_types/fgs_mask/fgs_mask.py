import logging

import numpy as np

from wfi_reference_pipeline.resources.wfi_meta_fgs_mask import WFIMetaFGSMask
from ..reference_type import ReferenceType

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
            bit_mask=None,
            outfile="roman_fgs_mask.asdf", # SAPP TODO - Verify if needed
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
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
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
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
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
