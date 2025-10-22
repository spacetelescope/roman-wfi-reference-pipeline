from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.darkdecaysignal.darkdecaysignal import (
    DarkDecaySignal,
    get_darkdecay_values_from_config,
)

# Base output folder (update as needed)
output_dir = "./dark_decay_refs"

# Create the MakeDevMeta object
tmp = MakeDevMeta(ref_type='DARKDECAY')

for i in range(1, 19):
    detector_id = f"WFI{i:02d}"
    
    # Update meta detector
    tmp.meta_darkdecay.instrument_detector = detector_id
    
    # Get amplitude and decay from config
    amp_decay = get_darkdecay_values_from_config(detector_id)
    
    # Create output filename
    outfile = f"{output_dir}/roman_dark_decay_{detector_id}.asdf"
    
    # Instantiate DarkDecaySignal
    dark_decay_ref = DarkDecaySignal(
        meta_data=tmp.meta_darkdecay,
        ref_type_data=amp_decay,
        outfile=outfile,
        clobber=True
    )
    
    # Optionally, save the ASDF file
    dark_decay_ref.save_dark_decay_signal_file()
    
    print(f"Created DarkDecaySignal reference for {detector_id} -> {outfile}")