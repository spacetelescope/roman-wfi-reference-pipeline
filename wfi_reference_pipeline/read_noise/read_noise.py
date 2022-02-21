import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats


class ReadNoise(ReferenceFile):
    """
    Class ReadNoise() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_read_noise() creates the asdf readnoise file.
    """

    #def __init__(self, zero1, zero2):

        #self.zero1 = zero1
        #self.zero2 = zero2

        # Future arrays
        #self.read_noise = None

    def __init__(self, readnoise_image, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_readnoise.asdf'

        # Access methods of base class ReferenceFile
        super(ReadNoise, self).__init__(readnoise_image, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with read noise file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI read noise reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'READNOISE'
        else:
            pass

    def get_read_noise(self):

        delta = self.zero1 - self.zero2
        _, noise, _ = sigma_clipped_stats(delta, sigma=5, maxiters=2, axis=1)
        variance = noise**2

        # Linear fit to the data. Could get fancy with this later to do
        # outlier rejection...
        var_func = np.poly1d(np.polyfit(range(delta.size[1]), variance[::-1] / 2, 1))
        self.read_noise = np.sqrt(var_func(0))

    def make_read_noise(self):
        """
        The method make_read_noise() generates a readnoise asdf file.

        Parameters
        ----------

        Outputs
        -------
        af: asdf file tree: {meta, data}
            meta:
            data: self
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the read noise object from the data model.
        rnfile         = rds.ReadnoiseRef()
        rnfile['meta'] = self.meta
        rnfile['data'] = self.data
        # Readnoise files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af      = asdf.AsdfFile()
        af.tree = {'roman': rnfile}
        af.write_to(self.outfile)
