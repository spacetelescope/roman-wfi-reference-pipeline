from pathlib import Path

from dask.distributed import Client

from wfi_reference_pipeline.reference_types.empirical_psf.empirical_psf import (
    EmpiricalPSF,
)
from wfi_reference_pipeline.reference_types.empirical_psf.psf_lib_generator import (
    generate_wfi_psf_library,
)
from wfi_reference_pipeline.resources.make_meta import MakeMeta

FILTERS = [
    "F062", "F087", "F106", "F129",
    "F146", "F158", "F184", "F213",
]

PIXEL_X = [
    4.0, 2047.5, 4091.0,
    4.0, 2047.5, 4091.0,
    4.0, 2047.5, 4091.0,
]

PIXEL_Y = [
    4.0, 4.0, 4.0,
    2047.5, 2047.5, 2047.5,
    4091.0, 4091.0, 4091.0,
]


def process_detector_filter(detector, optical_element):
    """
    Generate the PSF object library per detector and filter
    """
    meta = MakeMeta(ref_type="EPSF", instrument_detector="WFI01").meta_epsf

    meta.instrument_detector = f"WFI{detector:02d}"
    meta.optical_element = optical_element

    meta.pixel_x = PIXEL_X
    meta.pixel_y = PIXEL_Y

    result = generate_wfi_psf_library(
        detector=detector,
        optical_element=optical_element,
        pixel_x=PIXEL_X,
        pixel_y=PIXEL_Y,
        spectral_types=["A0V", "G2V", "M5V"],
        temperatures=[9550, 5778, 2900],
        logg_values=[3.95, 4.44, 5.0],
        defocus_values=[0, 1, 2],
        oversample=4,
        include_extended_psf=True,
        include_noipc=True,
    )

    output_dir = Path("/grp/roman/RFP/DEV/scratch")
    outfile = output_dir / (
        f"wfi{detector:02d}_"
        f"{optical_element.lower()}_epsf.asdf"
    )

    epsf = EmpiricalPSF(
        meta_data=meta,
        psf=result["psf"],
        extended_psf=result["extended_psf"],
        psf_noipc=result["psf_noipc"],
        extended_psf_noipc=result["extended_psf_noipc"],
        outfile=outfile,
        clobber=True,
    )

    return epsf, epsf.generate_outfile()


# ============================================================
# MAIN DRIVER
# ============================================================
def main(n_workers=0):

    detectors = range(1, 2) # change to do all detectors when ready
    # and then parallelize
    # --------------------------------------------------------
    # SERIAL MODE (n_workers = 0)
    # --------------------------------------------------------
    if n_workers == 0:

        print("Running in SERIAL mode (no Dask).")

        for detector in detectors:
            for optical_element in FILTERS:

                print(
                    f"Processing WFI{detector:02d} "
                    f"{optical_element}"
                )

                process_detector_filter(
                    detector,
                    optical_element,
                )

        return

    # --------------------------------------------------------
    # INVALID CONFIG (n_workers == 1)
    # --------------------------------------------------------
    if n_workers == 1:
        raise ValueError(
            "n_workers=1 is not allowed because it does not "
            "provide meaningful parallelization. "
            "Use n_workers=0 for serial execution or "
            "n_workers=40 (or similar) for parallel execution."
        )

    # --------------------------------------------------------
    # PARALLEL MODE
    # --------------------------------------------------------
    print(f"Running in PARALLEL mode with {n_workers} workers.")
    client = Client(n_workers=n_workers)
    print(f"Dask dashboard: {client.dashboard_link}")
    tasks = []

    for detector in detectors:
        for optical_element in FILTERS:

            tasks.append(
                client.submit(
                    process_detector_filter,
                    detector,
                    optical_element,
                )
            )
    client.gather(tasks)
    client.close()


if __name__ == "__main__":

    # default mode is serial and not parallelized
    main(n_workers=0)
