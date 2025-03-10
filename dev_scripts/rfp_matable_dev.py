from wfi_reference_pipeline.reference_types.multiaccumulationtable.multiaccumulationtable import MultiAccumulationTable
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

outfile = '/grp/roman/RFP/DEV/scratch/rfp_matable_dev_file.asdf'

input_file = '/Users/wschultz/RTB/prd-delivery-utils/src/ma_table_config_20241206.yaml'
# Use dev meta maker for DARK
tmp = MakeDevMeta(ref_type='MATABLE')

mat = MultiAccumulationTable(meta_data=tmp.meta_matable, file_list=[input_file], outfile=outfile)

mat.save_multi_accumulation_table()