"""
A base class for dealing with reference files.
"""

from ..utilities import basic_mask
from astropy.time import Time
import numpy as np
import os
import yaml

PIPELINE_VERSION = '0.0.1'


class ReferenceFile:

    def __init__(self, data, meta_data, bit_mask=None, clobber=False):

        self.data = data
        if bit_mask:
            self.mask = bit_mask.astype(np.uint32)
        else:
            self.mask = basic_mask.make_mask()

        # Grab the meta data from the yaml file.
        if type(meta_data) is dict:
            self.meta = meta_data
        else:
            with open(meta_data) as md:
                self.meta = yaml.safe_load(md)

        # Convert useafter date to Astropy.Time object. Update meta data
        # with a few constants.
        self.meta['useafter'] = Time(self.meta['useafter'])
        self.meta['author'] = f'WFI Reference File Pipeline '\
                              f'version {PIPELINE_VERSION}'
        self.meta['origin'] = 'STScI'
        self.meta['telescope'] = 'ROMAN'

        # Other stuff.
        self.clobber = clobber

    def check_output_file(self, outfile):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(outfile):
            if self.clobber:
                os.remove(outfile)
            else:
                raise FileExistsError(f'{outfile} already exists, and '
                                      f'clobber={self.clobber}!')
