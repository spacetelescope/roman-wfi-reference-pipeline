import logging
from pathlib import Path

import roman_datamodels as rdm

from romancal.pipeline.exposure_pipeline import ExposurePipeline
from wfi_reference_pipeline.constants import REF_TYPE_FLAT
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.flat.flat import Flat
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# from wfi_reference_pipeline.utilities.logging_functions import log_info

import sys
from concurrent.futures import ProcessPoolExecutor

def robust_process_map(fn, iterable, max_workers=None, chunksize=1, desc=None):
    """
    Executes a function across an iterable using multiprocessing.
    Uses tqdm.contrib.concurrent.process_map if available for progress tracking,
    otherwise falls back to standard concurrent.futures.ProcessPoolExecutor.

    Parameters
    ----------
    fn : callable
        The function to apply to each element of the iterable. It must be a 
        top-level function (picklable) to work with multiprocessing pools.
    iterable : iterable
        An iterable sequence of items (e.g., list, tuple, generator) to pass into 
        the target function.
    max_workers : int, optional
        The maximum number of worker processes to spawn. If None or omitted, it 
        defaults to the number of processors available on the machine.
    chunksize : int, optional
        The size of chunks into which the iterable is divided when submitting 
        tasks to worker processes. Increasing this value beyond 1 can drastically 
        improve performance for massive datasets by reducing IPC overhead.
    desc : str, optional
        A string prefix displayed before the progress bar or fallback logging statements, 
        handy for naming or categorizing the running operation.

    Returns
    -------
    list
        A list of results obtained from mapping `fn` over `iterable` in the original sequence.
    """
    try:
        from tqdm.contrib.concurrent import process_map
        return process_map(fn, iterable, max_workers=max_workers, chunksize=chunksize, desc=desc)
        
    except ImportError:
        if desc:
            print(f"[{desc}] running (tqdm not available)...", file=sys.stderr)
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(fn, iterable, chunksize=chunksize))

def prep_single(fname):
    """
    Auxiliary function to check an process one single file
    
    Parameters:
    -----------
    fname : str,
        Path to uncal L1 data to process.

    Returns:
    --------
    prep_output_file_path: str,
        Path to RFP ready-to-ingest processed image.
    """
    in_file = rdm.open(fname)

    # Skip these by default
    elp_kwargs =  {"steps": {
                   "source_catalog": {"skip": True},
                   "tweakreg": {"skip": True},
                   "flatfield": {"skip": True},
                   "photom": {"skip": True},
                  }

    # Check if any of the steps are done so they can be skipped 
    for key in in_file.meta.cal_step.keys():
        if in_file.meta.cal_step[key] == "COMPLETE":
            elp_kwargs["steps"][key] = {"skip": True}
        # Only process / use L1 images that have not been flat fielded
    if in_file.meta_cal_step["flatfield"] == "INCOMPLETE":
        result = ExposureLevelPipeline.call(in_file, **kwargs)
        prep_output_file_path = self.file_handler.format_prep_output_file_path(
            result.meta.filename)
        result.save(path=prep_output_file_path)
    else:
        prep_output_file_path = None
    return prep_output_file_path

class FlatPipeline(Pipeline):
    """
    Derived from Pipeline Base Class
    This is the entry point for all Dark Pipeline functionality

    Gives user access to:
    select_uncal_files : Selecting level 1 uncalibrated asdf files with input generated from config
    prep_pipeline : Preparing the pipeline using romancal routines and save outputs to go into superdark
    run_pipeline: Process the data and create new calibration asdf file for CRDS delivery
    restart_pipeline: (derived from Pipeline) Run all steps from scratch

    Usage:
    flat_pipeline = FlatPipeline("<detector string>")
    flat_pipeline.select_uncal_files()
    flat_pipeline.prep_pipeline()
    flat_pipeline.run_pipeline()
    flat_pipeline.pre_deliver()
    flat_pipeline.deliver()

    or

    flat_pipeline.restart_pipeline()

    """

    def __init__(self, detector):
        # Initialize baseclass from here for access to this class name
        super().__init__(REF_TYPE_FLAT, detector)
        self.flat_file = None

    # @log_info
    def select_uncal_files(self):
        self.uncal_files.clear()
        logging.info("FLAT SELECT_UNCAL_FILES")

        """ TODO THIS MUST BE REPLACED WITH ACTUAL SELECTION LOGIC USING PARAMS
        FROM CONFIG IN CONJUNCTION WITH HOW WE WILL OBTAIN INFORMATION FROM DAAPI """
        # Get files from input directory
        files = list(
            self.ingest_path.glob("*_uncal.asdf")  # Change this
        )

        self.uncal_files = files
        logging.info(f"Ingesting {len(files)} Files: {files}")

    # @log_info
    def prep_pipeline(self, file_list=None):
        """Prepare calibration data files by running data
        through select romancal steps"""
        logging.info("FLAT PREP")

        # Clean up previous runs
        self.prepped_files.clear()
        self.file_handler.remove_existing_prepped_files_for_ref_type()

        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.uncal_files

        for file in file_list:
            logging.info("OPENING - " + file.name)
            _aux = prep_single(file.name)
            if _aux is not None:
                self.prepped_files.append(_aux)

        logging.info(
            "Finished PREPPING files to make FLAT reference file from RFP")

        logging.info("Starting to make FLAT from PREPPED FLAT asdf files")

    # @log_info
    def run_pipeline(self, file_list=None):
        logging.info("FLAT PIPE")

        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            if self.prepped_files is not None:
                file_list = self.prepped_files
            else:
                raise ValueError(
                    "Prepare file or pass a (pre-processed) file list")

        tmp = MakeDevMeta(ref_type=self.ref_type)
        out_file_path = self.file_handler.format_pipeline_output_file_path(
            tmp.meta_flat.optical_element,
            tmp.meta_flat.instrument_detector,
        )

        rfp_flat = Flat(
            meta_data=tmp.meta_flat,
            file_list=file_list,
            ref_type_data=None,
            outfile=out_file_path,
            clobber=True,
        )
        _, _ = rfp_flat.make_flat_from_files(calc_error=True, lo=10, hi=1000)
        rfp_flat.populate_datamodel_tree()
        rfp_flat.generate_outfile()
        self.flat_file = rfp_flat
        logging.info("Finished RFP to make FLAT")

    def deliver(self):
        pass

    def pre_deliver(self):
        pass

    def prep_parallel(self, file_list=None, max_workers=4):
        logging.info(f"FLAT PREP -- using {max_workers} workers")
        # Convert file_list to a list of Path type files
        if file_list is not None:
            file_list = list(map(Path, file_list))
        else:
            file_list = self.uncal_files
        # Process the files in parallel using max_workers
        _aux = robust_process_map(prep_single, file_list, max_workers=max_workers, chunksize=1)

        for _name in _aux:
            if _name is not None:
                self.prepped_files.append(_name)
        logging.info("Finished PREP")

