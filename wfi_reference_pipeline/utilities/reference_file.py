from astropy.time import Time
import os
import yaml
import numpy as np

PIPELINE_VERSION = '0.0.1' # set global variable for pipeline version


class ReferenceFile:
    """
    Base class ReferenceFile() writes static meta data for all reference file types
    are written.
    """

    def __init__(self, data, meta_data, bit_mask=None, clobber=False):

        self.data = data

        if np.shape(bit_mask):
            print("Mask provided. Skipping internal mask generation.")
            self.mask = bit_mask.astype(np.uint32)
        else:
            self.mask = np.zeros((4096, 4096), dtype=np.uint32)

        # Grab the meta data from the yaml file if provided.
        if type(meta_data) is dict:
            self.meta = meta_data
        else:
            with open(meta_data) as md:
                self.meta = yaml.safe_load(md)

        # Convert use after date to Astropy.Time object.
        self.meta['useafter'] = Time(self.meta['useafter'])
        # Write static meta data for all file type.
        self.meta['author'] = f'WFI Reference File Pipeline version {PIPELINE_VERSION}'
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
                raise FileExistsError(f'{outfile} already exists, and clobber={self.clobber}!')
