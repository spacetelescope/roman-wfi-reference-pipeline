from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities import simulate_reads
from wfi_reference_pipeline.reference_types.flat.flat import Flat
import numpy as np

print('Dev to make Flat from a rate image.')
outfile = '/grp/roman/RFP/DEV/scratch/rfp_flat_dev_file.asdf'
# Use dev meta to instantiate rfp flat object.
tmp = MakeDevMeta(ref_type='FLAT')
# Example how to change the useafter in the meta data.
tmp.meta_flat.use_after = '2024-01-01T00:00:00.000'
# Simulate a cube of flat reads.
sim_dev_cube, sim_dev_rate_image = simulate_reads.simulate_flat_reads(10)
# flatten rate image
dev_flat_rate_image = sim_dev_rate_image / np.mean(sim_dev_rate_image)
# Instantiate rfp flat object.
rfp_flat = Flat(meta_data=tmp.meta_flat,
                ref_type_data=dev_flat_rate_image,
                outfile=outfile,
                clobber=True)
# Must populate error array.
rfp_flat.calculate_error()
# Must update data quality array.
rfp_flat.update_data_quality_array()
# Save file.
rfp_flat.generate_outfile()

