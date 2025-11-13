from wfi_reference_pipeline.reference_types.exposure_time_calculator.exposure_time_calculator import (
    ExposureTimeCalculator,
    update_etc_form_from_crds,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# First step in this process is to update the yml form in the repo or write it somewhere else locally. 
update_etc_form_from_crds()

# Once the yml form has been updated properly from the files on CRDS from the latest context one
# can start making etc asdf files ready for CRDS delivery

for det in range(1, 19):
    # Create meta data object for ETC ref file
    tmp = MakeDevMeta(ref_type='ETC')
    # Update meta per detector to get the right values from the form and update description
    tmp.meta_etc.instrument_detector = f"WFI{det:02d}"
    tmp.meta_etc.description = 'To support new ETC ref file creation by Rick Cosentino to help the ETC team use CRDS.'
    # Update the file name to match the detector
    fl_name = 'new_roman_etc_' + tmp.meta_etc.instrument_detector 
    # Instantiate an object and write the file out
    rfp_etc = ExposureTimeCalculator(meta_data=tmp.meta_etc,
                                     outfile=fl_name+'.asdf',
                                     clobber=True)
    rfp_etc.generate_outfile()
    
