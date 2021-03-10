from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from romancal.datamodels.reference_files.flat import FlatModel
from ..utilities import basic_mask
import numpy as np
import os
import yaml


class Flat:

    def __init__(self, ramp_image, yaml_meta, bit_mask=None, outfile=None, clobber=True):

        self.flat_image = ramp_image
        if bit_mask:
            self.flat_dq = bit_mask.astype(np.uint32)
        else:
            self.flat_dq = basic_mask.make_mask()

        # Load the metadata. Convert the useafter from a string to
        # an astropy.Time object.
        with open(yaml_meta) as ym:
            self.meta = yaml.safe_load(ym)
        self.meta['meta']['useafter'] = Time(self.meta['meta']['useafter'])

        # If no output file name is provided, make one.
        if outfile:
            self.outfile = outfile
        else:
            self.outfile = f"roman_{self.meta['meta']['instrument']['detector'].lower()}_" \
                           f"{self.meta['meta']['instrument']['optical_element'].lower()}_flat.asdf"

        # Check if file exists, and handle it appropriately.
        if os.path.exists(self.outfile):
            if clobber:
                os.remove(self.outfile)
            else:
                raise FileExistsError(f'{self.outfile} already exists, and clobber={clobber}!')

    def make_flat(self, low_qe_threshold=0.2, low_qe_bit=13):

        # Normalize the flat_image by the sigma-clipped mean.
        mean, _, _ = sigma_clipped_stats(self.flat_image)
        self.flat_image /= mean

        # Add DQ flag for low QE pixels.
        low_qe = np.where(self.flat_image < low_qe_threshold)
        self.flat_dq[low_qe] += 2**low_qe_bit

        # Construct the flat field object from the data model.
        flat_asdf = FlatModel(data=self.flat_image, err=np.zeros(self.flat_image.shape, dtype=np.float32),
                              dq=self.flat_dq)

        # Add in the meta data and history to the ASDF tree.
        for key, value in self.meta['meta'].items():
            flat_asdf.meta[key] = value
        flat_asdf.history = self.meta['history']

        # Write out the flat field ASDF file.
        flat_asdf.save(self.outfile)
