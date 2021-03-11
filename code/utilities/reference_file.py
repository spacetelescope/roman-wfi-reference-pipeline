"""
A base class for dealing with reference files.
"""

from ..utilities import basic_mask
from astropy.time import Time
import numpy as np
import os
import yaml

PIPELINE_VERSION = '0.0'


class ReferenceFile:

    def __init__(self, data, meta_data, bit_mask=None, clobber=False):

        self.data = data
        if bit_mask:
            self.mask = bit_mask.astype(np.uint32)
        else:
            self.mask = basic_mask.make_mask()

        # Grab the meta data from the yaml file.
        with open(meta_data) as md:
            self.meta = yaml.safe_load(md)

        # Convert useafter date to Astropy.Time object. Update meta data
        # with a few constants.
        self.meta['meta']['useafter'] = Time(self.meta['meta']['useafter'])
        self.meta['meta']['author'] = f'WFI Reference File Pipeline version {PIPELINE_VERSION}'

        # Other stuff.
        self.clobber = clobber

    def check_output_file(self, outfile):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(outfile):
            if self.clobber:
                os.remove(outfile)
            else:
                raise FileExistsError(f'{outfile} already exists, and clobber={self.clobber}!')
