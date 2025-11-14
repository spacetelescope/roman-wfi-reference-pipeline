from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.dark_decay_signal.dark_decay_signal import DarkDecaySignal

# Base output folder (update as needed)
output_dir = "/grp/roman/RFP/DEV/build_files/Build_26Q1_B20"

# Create the MakeDevMeta object
tmp = MakeDevMeta(ref_type='DARKDECAY')

for i in range(1, 19):
    detector_id = f"WFI{i:02d}"
    
    # Update meta detector
    tmp.meta_darkdecay.instrument_detector = detector_id
    
    # Get amplitude and decay from config
    amp_decay = get_darkdecay_values_from_config(detector_id)
    print(amp_decay, detector_id)

    # Create output filename
    outfile = f"{output_dir}/roman_dark_decay_{detector_id}.asdf"
    
    rfp_dark_decay = DarkDecaySignal(
        meta_data=tmp.meta_darkdecay,
        ref_type_data=amp_decay,
        outfile=outfile,
        clobber=True
    )

    rfp_dark_decay.save_dark_decay_signal_file()
    # Default compression was around 1000x - not bad
    print(f"Created DarkDecaySignal reference for {detector_id} -> {outfile}")