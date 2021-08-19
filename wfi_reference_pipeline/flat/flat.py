import asdf
from astropy.stats import sigma_clipped_stats
import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import numpy as np


class Flat(ReferenceFile):

    def __init__(self, ramp_image, meta_data, bit_mask=None, outfile=None,
                 clobber=False):
        # If no output file name given, just set one now.
        self.outfile = outfile if outfile else 'roman_flat.asdf'

        super(Flat, self).__init__(ramp_image, meta_data, bit_mask=bit_mask,
                                   clobber=clobber)

        # Update metadata with constants.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Flat field file.'
        else:
            pass

    def make_flat(self, low_qe_threshold=0.2, low_qe_bit=13):
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Normalize the flat_image by the sigma-clipped mean.
        mean, _, _ = sigma_clipped_stats(self.data)
        self.data /= mean

        # Add DQ flag for low QE pixels.
        low_qe = np.where(self.data < low_qe_threshold)
        self.mask[low_qe] += 2 ** low_qe_bit

        # Construct the flat field object from the data model.
        flatfile = rds.FlatRef()
        flatfile['meta'] = self.meta
        flatfile['data'] = self.data
        flatfile['dq'] = self.mask
        flatfile['err'] = np.zeros(self.data.shape, dtype=np.float32)

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': flatfile}
        af.write_to(self.outfile)
        #for key, value in self.meta['meta'].items():
        #    flat_asdf.meta[key] = value
        #flat_asdf.history = self.meta['history']

        # Write out the flat field ASDF file.
        #flat_asdf.save(self.outfile)
