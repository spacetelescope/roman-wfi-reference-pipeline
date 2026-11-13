import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import roman_datamodels as rdm
from astropy.io import fits
from romancal.dark_decay import DarkDecayStep
from romancal.dq_init import DQInitStep
from romancal.linearity import LinearityStep
from romancal.ramp_fitting import ramp_fit_step
from romancal.refpix import RefPixStep
from romancal.saturation import SaturationStep
from romancal.wfi18_transient import WFI18TransientStep

from wfi_reference_pipeline.config.config_access import get_pipelines_config
from wfi_reference_pipeline.constants import (
    REF_TYPE_FGS_MASK,
)
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.fgs_mask.fgs_mask import FGSMask
from wfi_reference_pipeline.resources.make_meta import MakeMeta


class FGSMaskPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Fine Guiding System (FGS) Mask Pipeline functionality.

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save output
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    fgs_mask_pipeline = FGSMaskPipeline("<detector string>")
    fgs_mask_pipeline.select_uncal_files()
    fgs_mask_pipeline.prep_pipeline()
    fgs_mask_pipeline.run_pipeline()
    fgs_mask_pipeline.pre_deliver()
    fgs_mask_pipeline.deliver()

    or

    fgs_mask_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FGS_MASK, detector)
        self.config = get_pipelines_config(REF_TYPE_FGS_MASK)

        self.flat_filelist = []
        self.dark_filelist = []

    def select_uncal_files(self):
        """Select the uncal files to be run through the RFP"""
        # Clearing from previous run
        self.uncal_files.clear()

        # TODO: how would users go about specifying which detector they want
        # to focus on? The paths here are specified in the config file so idk
        files = list(self.ingest_path.glob("*_uncal.asdf"))

        self.uncal_files = files

        logging.info(f"Ingesting {len(files)} files: {files}")


    def prep_pipeline(self, file_list=None):
        """
        Prepare calibration data files by running data through select romancal steps.
        Sort the filelist, then create the superdark and superflat from the calibrated files.
        """
        logging.info("FGS_MASK PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = list(map(Path, self.uncal_files))

        for file in file_list:

            logging.info("OPENING - " + file.name)
            perform_rampfit = False

            if "flat" in os.path.basename(file):
                perform_rampfit = True

            self._run_romancal(file, perform_rampfit=perform_rampfit)

        logging.info("Finished PREPPING files to make FGS_MASK reference file from RFP")

        logging.info("Sorting prepped files into darks and flats")
        self._sort_filelist()

        logging.info("All files prepped, ready to run FGSMask")
        

    def run_pipeline(self, input_super_dark=None, input_super_rate=None):
        """
        Run the FGS workflow on the flat and dark files to create a DQ bitmask
        using the FGSFlags in fgs_mask.py.
        """
        logging.info("FGS_MASK PIPE")

        tmp = MakeMeta(ref_type=self.ref_type, instrument_detector=self.detector)

        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_fgs_mask.mode,
            tmp.meta_fgs_mask.instrument_detector,
        )

        self.rfp_fgs_mask = FGSMask(
            meta_data=tmp.meta_fgs_mask,
            dark_filelist=self.dark_filelist,
            flat_filelist=self.flat_filelist,
            input_super_dark=input_super_dark,
            input_super_rate=input_super_rate,
            outfile=out_file_path,
            clobber=True,
        )

        # Running the actual FGS mask algorithms
        self.rfp_fgs_mask.make_fgs_mask_image(
            dead_sigma_thr=self.config["DEAD_SIGMA_THR"],
            hot_thr=self.config["HOT_THR"],
            superhot_thr=self.config["SUPERHOT_THR"],
            high_cds_thr=self.config["HIGH_CDS_THR"],
            low_qe_thr=self.config["LOW_QE_THR"],
            bad_flat_thr=self.config["BAD_FLAT_THR"]
        )

        self.rfp_fgs_mask.generate_outfile()

        logging.info("Finished RFP to make FGS_MASK")


    def pre_deliver(self, file_change_note=None):
        """
        The final FGS BPM that is deliverd to PSS and then sent to the MOC is in a different
        format than the science bad pixel mask on CRDS (see docstring of `convert_mask_to_pss_format`).
        This function converts the new bitmask to the file format used by the Roman Planning and Scheduling Subsystem (PSS).

        NOTE: This is the format that will be delivered to PSS.

        Parameters
        ----------
        file_change_note : str, optional
            String that indicates if any major changes are associated with the creation
            of this mask. Default is None.
        """
        self.convert_mask_to_pss_format()
        self.save_pss_mask(file_change_note=file_change_note)


    def deliver(self):
        """
        FGS masks are *not* delivered to CRDS; they follow a different (manual) delivery process using central store.
        """
        pass

    
    def convert_mask_to_pss_format(self):
        """
        The final FGS BPM that is deliverd to PSS and then sent to the MOC is in a different
        format than the science bad pixel mask on CRDS. It is in detector coordinates
        (rather than the SOC's science coordinates), and it is a boolean mask (instead
        of np.uint32 bitvalues).
        
        This function puts the DQ mask into detector coordinates, then converts it to boolean.
        The converted PSS mask is then set as `self.mask_pss`.
        """
        mask_det_coords = self._change_coord_to_det(self.rfp_fgs_mask.mask_image)
        self.mask_pss = self._convert_mask_to_boolean(mask_det_coords)


    # TODO: Is it the standard for the default value to be explicitly set in 
    # subsequent functions? or just the first time it's used? (file_change_note)
    def save_pss_mask(self, file_change_note):
        """
        Using the PSS mask, create the FITS file with the correct header.
        The outpath of the new mask file is set as `self.pss_mask_outpath`.
        """

        mask_header = self._make_fits_header(file_change_note=file_change_note)
        hdu = fits.PrimaryHDU(data=self.mask_pss,
                              header=mask_header)
        
        self.pss_mask_outpath = os.path.join(self.file_handler.pipeline_out_path,
                                             f"fgs_mask_{self.detector}.fits")
    
        hdu.writeto(self.pss_mask_outpath,
                    overwrite=True)


    def _make_fits_header(self, file_change_note):
        """
        Create the FITS header for the PSS mask.
        Example of header values:
            hdr['DETECTOR'] = "WFI01"
            hdr['AUTHOR'] = "Nancy Grace Roman"
            hdr['DATETIME'] = 2026-05-01 # datetime in str format
            hdr['NBADPIX'] = 20000
            hdr['CHANGE_NOTE'] = "First mask using new dead pixel identification algorithm"
        """
        hdr = fits.Header()

        hdr['DETECTOR'] = (self.detector, 'WFI detector number')
        hdr['AUTHOR']   = (self.rfp_fgs_mask.meta_data.author, 'Author of file')
        hdr['DATETIME'] = (datetime.now().strftime('%Y-%m-%d'), 'Date of file creation')
        hdr['NBADPIX']  = (int(np.count_nonzeros(self.mask_pss)), 'Number of bad pixels')
        hdr['CHANGE_NOTE'] = (file_change_note, 'High-level changes to mask derivation')

        return hdr
        

    def _convert_mask_to_boolean(self, mask):
        """Return a mask where any non-zero value = 1"""
        return (mask != 0).astype("uint8")


    def _change_coord_to_det(self, arr):
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

        position = detector_pos[self.detector]

        if position == "lower right":
            return arr[:, ::-1]

        else:
            return arr[::-1]


    def restart_pipeline(self):

        self.select_uncal_files()
        self.prep_pipeline()
        self.run_pipeline()
        self.pre_deliver()
        self.deliver()

        return

    
    def _run_romancal(self, file, perform_rampfit=False):
        """
        Run romancal on a single file. Add the output filepath to `self.prepped_files`.

        Parameters:
        -----------
        file : str
            The L1 file to be run through romancal.
        perform_rampfit : bool, default=False
            If True, run all steps of romancal up to the ramp fitting step.
        """
        with rdm.open(file) as f:

            result = DQInitStep.call(f, save_results=False)
            result = SaturationStep.call(result, save_results=False)
            result = RefPixStep.call(result, save_results=False)

            if perform_rampfit:
                result = DarkDecayStep.call(result, save_results=False)
                result = WFI18TransientStep.call(result, save_results=False)
                result = LinearityStep.call(result, save_results=False)
                result = ramp_fit_step.RampFitStep.call(result, save_results=False)

            prep_output_file_path = self.file_handler.format_prep_output_file_path(
                result.meta.filename
            )
            result.save(path=prep_output_file_path)

            self.prepped_files.append(prep_output_file_path)

    
    def _sort_filelist(self):
        """
        Sort the prepped files into flats and darks based on filename.
        """
        logging.info("Sorting the files into flats vs darks in self.file_list")

        invalid_files = []

        for file in self.prepped_files:
            filename = os.path.basename(file).lower()

            if "flat" in filename:
                self.flat_filelist.append(file)

            elif "dark" in filename:
                self.dark_filelist.append(file)

            else:
                invalid_files.append(file)

        if invalid_files:
            # TODO: should we set this as an attr instead of raising an error?
            raise ValueError("The following files can not be sorted in prepped flats or darks:", invalid_files)
