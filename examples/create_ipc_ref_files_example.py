import os
from astropy.time import Time

from wfi_reference_pipeline.reference_types.inter_pixel_capacitance.inter_pixel_capacitance import InterPixelCapacitance
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.constants import WFI_DETECTORS

print(f"Generating {len(WFI_DETECTORS)} IPC ASDF files using the RFP...\n")

for detector in WFI_DETECTORS:

    # Create metadata object
    tmp = MakeDevMeta(ref_type="IPC")

    # Update metadata fields
    tmp.meta_ipc.instrument_detector = detector
    tmp.meta_ipc.description = "Cillis-derived TVAC IPC kernels for Roman WFI SOC implementation."
    tmp.meta_ipc.use_after = Time("2023-03-01T00:00:00.000")

    # Optional but recommended (schema compliance)
    tmp.meta_ipc.reftype = "IPC"

    # Create unique outfile per detector
    outfile = f"roman_ipc_{detector}.asdf"

    # Instantiate reference file object
    rfp_ipc = InterPixelCapacitance(
        meta_data=tmp.meta_ipc,
        outfile=outfile,
        clobber=True
    )

    # Generate file
    rfp_ipc.generate_outfile()

    print(f"Created: {outfile}")

print("\nAll IPC reference files generated successfully.")