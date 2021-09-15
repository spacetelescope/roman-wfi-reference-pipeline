"""
A base class for dealing with reference files.
"""

from ..utilities import basic_mask
import numpy as np
from astropy.time import Time
import os
import yaml

PIPELINE_VERSION = '0.0.1' # set global variable for pipeline version


class ReferenceFile:
    """ The child class ReferenceFile() gives writes statis meta data to each reference
    file type within the parent classes.
    """

    def __init__(self, data, meta_data, bit_mask=None, clobber=False):

        self.data = data
        if bit_mask:
            self.mask = bit_mask.astype(np.uint32)
        else:
            self.mask = basic_mask.make_mask()

        # Grab the meta data from the yaml file if provided.
        if type(meta_data) is dict:
            self.meta = meta_data
        else:
            with open(meta_data) as md:
                self.meta = yaml.safe_load(md)

        # Convert use after date to Astropy.Time object.
        self.meta['useafter']  = Time(self.meta['useafter'])
        # Write static meta data for all file type.
        self.meta['author']    = f'WFI Reference File Pipeline version {PIPELINE_VERSION}'
        self.meta['origin']    = 'STScI'
        self.meta['telescope'] = 'ROMAN'

        # Other stuff.
        self.clobber = clobber

    def check_output_file(self, outfile):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(outfile):
            if self.clobber:
                os.remove(outfile)
            else:
                raise FileExistsError(f'{outfile} already exists, and clobber={self.clobber}!')
