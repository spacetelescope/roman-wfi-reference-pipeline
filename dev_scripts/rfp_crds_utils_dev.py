from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.resources.make_ops_meta import MakeOpsMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

tmp = MakeDevMeta(ref_type='DARK')
sim_dev_cube, sim_dev_rate_image = simulate_dark_reads(5)
rfp_dark_wfi01 = Dark(tmp.meta_dark, ref_type_data=sim_dev_rate_image, get_crds_mask=True)


tmp2 = MakeOpsMeta(ref_type='DARK', instrument_detector='WFI07')
rfp_dark_wfi07 = Dark(tmp2.meta_dark, ref_type_data=sim_dev_rate_image, get_crds_mask=True)
