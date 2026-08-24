from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.distortion.distortion import Distortion
from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import PixelArea

import asdf
from astropy import units as u
from astropy.time import Time
from astropy.modeling import models


tmp = MakeDevMeta(ref_type="DISTORTION")
tmp.meta_distortion.author = "Richard G Cosentino"
tmp.meta_distortion.description = (
    "The Geometric Distortion reference file on Roman "
    "CRDS reflects newest changes to the pysiaf package "
    "corresponding to versions v0.27.0."
)
tmp.meta_distortion.useafter = Time(
    "2026-08-14T00:00:00.000",
    format="isot",
)
tmp.meta_distortion.pedigree = "GROUND"

tmp.meta_distortion.export_asdf_meta()

rfp_distortion = Distortion(meta_data=tmp.meta_distortion)
rfp_distortion.make_siaf_distortion(rfp_distortion.meta_data.instrument_detector)

