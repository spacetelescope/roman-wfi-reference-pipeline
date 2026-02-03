"""
This is to test the timing of the full Mask() module workflow
when supplied with both flats and darks.
"""
from wfi_reference_pipeline.constants import (
    REF_TYPE_MASK,
)
from wfi_reference_pipeline.pipelines.pipeline import Pipeline
from wfi_reference_pipeline.reference_types.mask.mask import Mask
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

import time
import glob
import os

import warnings
import logging

from asdf.exceptions import AsdfPackageVersionWarning

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="asdf"
)

warnings.filterwarnings(
    "ignore",
    category=AsdfPackageVersionWarning,
    module="asdf"
)

os.environ["ROMAN_VALIDATE"] = "false"

DET = "WFI09"
longdarks = glob.glob(f"irrc/*{DET}*")
flats = glob.glob(f"irrc_flats/*{DET}*")

tmp = MakeDevMeta(ref_type="MASK")

tmp.meta_mask.use_after = '2020-05-01T00:00:00.000'
tmp.meta_mask.author = "Sierra"
tmp.meta_mask.instrument_detector = DET
tmp.meta_mask.description = "Testing new mask from dark modules."

outpath = ""

rfp_mask = Mask(meta_data=tmp.meta_mask,
                file_list=longdarks,
                outfile=f"mask_test_{DET}.asdf",
                clobber=True)

rfp_mask.make_mask_image(from_smoothed=False,
                         flat_filelist=flats,
                         intermediate_path=outpath)

rfp_mask.generate_outfile()
