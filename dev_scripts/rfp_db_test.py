import argparse
import sys

from wfi_reference_pipeline.constants import (
    REF_TYPE_DARK,
    REF_TYPE_FLAT,
    REF_TYPE_MASK,
    REF_TYPE_READNOISE,
    REF_TYPE_REFPIX,
    WFI_DETECTORS,
    WFI_REF_TYPES,
)

from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.pipelines.flat_pipeline import FlatPipeline
from wfi_reference_pipeline.pipelines.mask_pipeline import MaskPipeline
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline
from wfi_reference_pipeline.pipelines.refpix_pipeline import RefPixPipeline


def main(args=None):
    """
    Run this file entry point for rfp_db_test file
    Accepts reference_type, detector, and output_file string for needed input parameters
    Runs ZERO steps of the wfi_reference_pipeline but will add to database.

    To use:
    `python rfp_db_test.py DARK WFI01 /path/to/output_file.asdf`
    """

    parser = argparse.ArgumentParser(
        description="Reference File Pipeline - Database Interaction Dev Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "reference_type",
        type=str.upper,
        choices=list(sorted([REF_TYPE_DARK,
                            REF_TYPE_FLAT,
                            REF_TYPE_MASK,
                            REF_TYPE_READNOISE,
                            REF_TYPE_REFPIX])),
        help="Reference Type to process",
    )
    parser.add_argument(
        "detector",
        type=str.upper,
        choices=list(sorted(WFI_DETECTORS)),
        help="WFI Detector Name to process",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output File to add to DB to process",
    )
    args = parser.parse_args()
    ref_type = args.reference_type
    detector = args.detector
    output_file = args.output_file


    if detector.upper() not in WFI_DETECTORS:
        raise KeyError(
            f"Invalid Detector {detector} - must be one of {WFI_DETECTORS}"
        )

    pipeline = None
    if ref_type == REF_TYPE_DARK:
        pipeline = DarkPipeline(detector)
    elif ref_type == REF_TYPE_FLAT:
        pipeline = FlatPipeline(detector)
    elif ref_type == REF_TYPE_MASK:
        pipeline = MaskPipeline(detector)
    elif ref_type == REF_TYPE_READNOISE:
        pipeline = ReadnoisePipeline(detector)
    elif ref_type == REF_TYPE_REFPIX:
        pipeline = RefPixPipeline(detector)
    else:
        raise KeyError(f"ref_type {ref_type} not currently utilized")

    if pipeline.db_handler:
        pipeline.db_handler.db_entry.rfp_log_pro.pipeline_cmd = "manual_test"
        pipeline.db_handler.db_entry.rfp_log_pro.output_filename = output_file
        pipeline.db_handler.update_db_entry()
    else:
        raise KeyError(f"ref_type {ref_type} not yet implemented")


if __name__ == "__main__":
    sys.exit(main())




