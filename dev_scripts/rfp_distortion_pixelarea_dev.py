from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.reference_types.distortion.distortion import Distortion
from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import PixelArea

import asdf
from astropy import units as u
from astropy.time import Time
from astropy.modeling import models

meta_distortion = {
    "author": "Richard G Cosentino",
    "description": (
        "The Geometric Distortion reference file on Roman "
        "CRDS reflects newest changes to the pysiaf package corresponding to versions v0.27.0."
    ),
    "input_units": u.pix,
    "instrument": {
        "detector": "WFI01",
        "name": "WFI",
        "optical_element": "F158",
        "p_optical_element": (
            "F062|F087|F106|F129|F146|F158|F184|F213|GRISM|PRISM|DARK|"
        ),
    },
    "origin": "STSCI",
    "output_units": u.arcsec,
    "pedigree": "GROUND",
    "reftype": "DISTORTION",
    "telescope": "ROMAN",
    "useafter": Time("2026-08-14T00:00:00.000", format="isot"),
}


print("The default metadata values are: ", tmp.meta_distortion)

rfp_distortion

