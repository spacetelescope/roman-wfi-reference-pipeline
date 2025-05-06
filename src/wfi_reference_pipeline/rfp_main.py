import argparse
import sys

from constants import (
    REF_TYPE_ABVEGAMAGNITUDEOFFSET,
    REF_TYPE_APERTURECORRECTION,
    REF_TYPE_DARK,
    REF_TYPE_DISTORTION,
    REF_TYPE_FLAT,
    REF_TYPE_GAIN,
    REF_TYPE_INVERSELINEARITY,
    REF_TYPE_IPC,
    REF_TYPE_LINEARITY,
    REF_TYPE_MASK,
    REF_TYPE_MULTIACCUMULATIONTABLE,
    REF_TYPE_PIXELAREA,
    REF_TYPE_READNOISE,
    REF_TYPE_REF_COMMON,
    REF_TYPE_REF_EXPOSURE_TYPE,
    REF_TYPE_REF_OPTICAL_ELEMENT,
    REF_TYPE_REFPIX,
    REF_TYPE_SATURATION,
    REF_TYPE_SUPERBIAS,
    REF_TYPE_WFI_IMG_PHOTOM,
    WFI_DETECTORS,
    WFI_DETECTORS_ALL,
    WFI_REF_TYPES,
)

from wfi_reference_pipeline.pipelines.dark_pipeline import DarkPipeline
from wfi_reference_pipeline.pipelines.flat_pipeline import FlatPipeline
from wfi_reference_pipeline.pipelines.mask_pipeline import MaskPipeline
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline
from wfi_reference_pipeline.pipelines.refpix_pipeline import RefPixPipeline


def main(arguments):
    """
    Main entry point for wfi_reference_pipeline project
    Accepts reference_type and detector
    Runs all steps of the wfi_reference_pipeline process for ref_type / detector combination.
    """

    parser = argparse.ArgumentParser(
        description="Roman Space Telescope - Reference File Pipeline main access point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "reference_type",
        type=str.upper,
        choices=list(sorted(WFI_REF_TYPES)),
        help="Reference Type to process",
    )
    parser.add_argument(
        "-d",
        "--detector",
        type=str.upper,
        choices=list(sorted(WFI_DETECTORS)),
        help="WFI Detector Name to process",
        default=WFI_DETECTORS_ALL,
    )
    args = parser.parse_args()
    ref_type = args.reference_type
    input_detector = args.detector

    # Note: CURRENTLY GOING TO RUN ONE DETECTOR AT A TIME.
    # IF PARALLELIZATION IS NEEDED ACROSS DETECTORS, WE CAN ACCOUNT FOR THAT LATER

    # need way to handle select_uncal_file (keep this routine as part of the pipeline)
    # information it will need are
    # TODO NOTES
    # Step 1 - DAAPI discussion with Andrew on how to receive daapi files in ingest directory
    # Step 2 - Update RFP DB with new files.
    # Step 3 - Check if criteria to make specific reference file is met

    detectors = []
    if input_detector.upper() == WFI_DETECTORS_ALL:
        detectors = [detector for detector in WFI_DETECTORS]
    elif input_detector.upper() in WFI_DETECTORS:
        detectors.append(input_detector.upper())
    else:
        raise KeyError(
            f"Invalid Detector {input_detector} - must be 'ALL' or one of {WFI_DETECTORS}"
        )

    pipeline = None
    for detector in detectors:
        if ref_type == REF_TYPE_ABVEGAMAGNITUDEOFFSET:
            pass
        elif ref_type == REF_TYPE_APERTURECORRECTION:
            pass
        elif ref_type == REF_TYPE_DARK:
            pipeline = DarkPipeline(detector)
        elif ref_type == REF_TYPE_DISTORTION:
            pass
        elif ref_type == REF_TYPE_FLAT:
            pipeline = FlatPipeline(detector)
        elif ref_type == REF_TYPE_GAIN:
            pass
        elif ref_type == REF_TYPE_INVERSELINEARITY:
            pass
        elif ref_type == REF_TYPE_IPC:
            pass
        elif ref_type == REF_TYPE_LINEARITY:
            pass
        elif ref_type == REF_TYPE_MASK:
            pipeline = MaskPipeline(detector)
        elif ref_type == REF_TYPE_MULTIACCUMULATIONTABLE:
            pass
        elif ref_type == REF_TYPE_PIXELAREA:
            pass
        elif ref_type == REF_TYPE_READNOISE:
            pipeline = ReadnoisePipeline(detector)
        elif ref_type == REF_TYPE_REF_COMMON:
            pass
        elif ref_type == REF_TYPE_REF_EXPOSURE_TYPE:
            pass
        elif ref_type == REF_TYPE_REF_OPTICAL_ELEMENT:
            pass
        elif ref_type == REF_TYPE_REFPIX:
            pipeline = RefPixPipeline(detector)
        elif ref_type == REF_TYPE_SATURATION:
            pass
        elif ref_type == REF_TYPE_SUPERBIAS:
            pass
        elif ref_type == REF_TYPE_WFI_IMG_PHOTOM:
            pass
        else:
            raise KeyError(f"ref_type {ref_type} not valid - try {WFI_REF_TYPES}")

        if pipeline:
            pipeline.restart_pipeline()
        else:
            raise KeyError(f"ref_type {ref_type} not yet implemented")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
