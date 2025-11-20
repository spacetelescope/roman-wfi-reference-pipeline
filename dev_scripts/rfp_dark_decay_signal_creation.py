from wfi_reference_pipeline.reference_types.dark_decay_signal.dark_decay_signal import (
    DarkDecaySignal,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

tmp = MakeDevMeta(ref_type='DARKDECAYSIGNAL')

rfp_dark_decay_signal = DarkDecaySignal(meta_data=tmp.meta_dark_decay_signal, clobber=True)
rfp_dark_decay_signal.generate_outfile()
