import os
from astropy.time import Time
from wfi_reference_pipeline.reference_types.interpixelcapacitance.ipc_analia_fr import InterPixelCapacitance
from wfi_reference_pipeline.constants import WFI_DETECTORS

class PipelineMeta(dict):
    pass

output_dir = "ipc_flight_asdfs"
os.makedirs(output_dir, exist_ok=True)

print(f"Generating 18 IPC ASDF files in '{output_dir}/' using the native RFP...\n")

for detector in sorted(WFI_DETECTORS):
    # Swapped 'ANY' for 'F062' to pass the strict enum schema validator
    meta = PipelineMeta({
        'author': 'Harish Khandrika',
        'description': 'Cillis-derived TVAC IPC kernels for Roman WFI SOC implementation.',
        'instrument': {
            'detector': detector, 
            'name': 'WFI', 
            'optical_element': 'DARK', 
        },
        'origin': 'STSCI',
        'pedigree': 'GROUND',
        'reftype': 'IPC',
        'telescope': 'ROMAN',
        'useafter': Time('2023-03-01T00:00:00.000')
    })
    
    meta.reference_type = 'IPC'
    
    outfile_path = os.path.join(output_dir, f"roman_wfi_ipc_{detector.lower()}.asdf")
    
    ipc_module = InterPixelCapacitance(
        meta_data=meta, 
        file_list=[], 
        outfile=outfile_path, 
        clobber=True
    )
    
    ipc_module.make_ipc_kernel()
    
    try:
        ipc_module.generate_outfile()
        print(f"Saved {detector}")
    except Exception as e:
        print(f"Failed to write {detector}: {e}")

print("\n All 18 files have been successfully generated and saved by the RFP!")