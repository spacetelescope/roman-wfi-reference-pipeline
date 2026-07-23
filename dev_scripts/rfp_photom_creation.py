from wfi_reference_pipeline.reference_types.photom.photom import Photom, build_gain_and_pam_dict_from_crds
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

# First create a MakeDevMeta object and update some metadata
tmp = MakeDevMeta(ref_type='PHOTOM')
print("The default metadata values are: ", tmp.meta_photom)

# Manually setting various metadata
tmp.meta_photom.use_after = '2020-05-01T00:00:00.000'
tmp.meta_photom.author = 'RFP'
tmp.meta_photom.pedigree = 'GROUND'

# Creating the Photom object
rfp_photom = Photom(meta_data=tmp.meta_photom, 
                     outfile=f'roman_photom_file_wfi{tmp.meta_photom.instrument_detector:02}.asdf',
                     clobber=True)

# Loop through all 18 detectors to output the ASDF files
for wfi_num in range(18):
    detector = wfi_num + 1
    # Update the detector number in the metadata
    tmp.meta_photom.instrument_detector = f'WFI{detector:02d}'

    # Build the photom dictionary
    rfp_photom.make_photom()
    
    # Generate the CRDS-compliant reference file
    rfp_photom.outfile = f'roman_photom_file_wfi{detector:02}.asdf'
    rfp_photom.generate_outfile()
