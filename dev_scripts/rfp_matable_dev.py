from wfi_reference_pipeline.reference_types.multiaccumulationtable.multiaccumulationtable import MultiAccumulationTable
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

outfile = '/grp/roman/RFP/DEV/scratch/rfp_matable_dev_file_revG_w_dms_tests.asdf'

input_file = '/Users/wschultz/RTB/prd-delivery-utils/src/rev_post-G/ma_table_ref_20250701_rev_post-G_dev.yaml'

tmp = MakeDevMeta(ref_type='MATABLE')
tmp.meta_matable.author = 'William C. Schultz'
tmp.meta_matable.description = "Implementation of Revision G tables and shifting the table numbers to match the flight paradigm. Revision F tables are left untouched for supporting previous MRTs. DMS testing tables 109 and 110 are also included. This reference file coincides with PRD release 20 and the PSS patch to update the MA tables.\nThe MA tables are agnostic of the detector and thus the value of the detector in the meta data (WFI01) should be ignored."
tmp.meta_matable.pedigree = "GROUND"

#Change the PRD version of the release
tmp.meta_matable.prd_version = 20 # June 25 2025

mat = MultiAccumulationTable(meta_data=tmp.meta_matable, file_list=[input_file], outfile=outfile, clobber=True)

mat.save_multi_accumulation_table()