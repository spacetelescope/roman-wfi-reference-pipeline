import logging
import asdf
import numpy as np
import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
# from ..utilities.logging_functions import log_info
from astropy import units as u
import pdb


# @log_info
class Gain(ReferenceFile):
    """
    Class Gain() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_gain() creates the asdf gain file.
    """

    def __init__(self, input_data, meta_data, bit_mask=None, outfile=None,
                 clobber=False):

        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_gain.asdf'

        # Access methods of base class ReferenceFile.
        super(Gain, self).__init__(input_data, meta_data, bit_mask=bit_mask,
                                   clobber=clobber)

        # Update metadata with gain file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI gain reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'GAIN'
        else:
            pass

        # if isinstance(self.data, list):
        #     self.data = np.array(self.data)
        # if self.data.shape[0] % 2:
        #     self.data = self.data[:-1]
        #     logging.warn('Odd number of input observations given. Removing the '
        #                  'last observation from analysis.')

        self.gain = input_data

    def make_gain(self):
        """
        The method make_gain() uses the photon transfer curve to estimate
        the gain in units of electrons/DN.

        Parameters
        ----------

        Outputs
        -------
        None
        """

        data_shape = self.data[0].shape
        n_pairs = len(self.data) // 2

        # Set up the variance and mean signal level arrays that
        # we'll need for the photon transfer curve. For a full WFI
        # SCA, these should be N x 4088 x 4088, where N is the number
        # of reads or resultants in the input data.
        var_arr = np.zeros(data_shape, dtype=np.float)
        signal_arr = np.zeros(data_shape, dtype=np.float)

        # For each read/resultant, step over each pair of flats and
        # compute the difference. Save the difference to a temporary
        # N x 4088 x 4088 array, where N is the number of pairs.
        #
        # From the temporary array, compute the variance and store it
        # in the plane of var_arr corresponding to a given read/resultant.
        for result in range(data_shape[0]):
            temp_arr = np.zeros((n_pairs, data_shape[1], data_shape[2]),
                                dtype=np.float)
            for p in range(n_pairs):
                temp_arr[p] = self.data[2 * p][result] - \
                              self.data[(2 * p) + 1][result]

            # Compute the variance.
            mean_diff = np.mean(temp_arr, axis=0)
            numerator = np.sum((temp_arr - mean_diff) ** 2, axis=0)
            var_arr[result] += (numerator / (2 * n_pairs))

        # Compute the mean signal per resultant.
        signal_arr = np.mean(self.data, axis=0)

        return signal_arr, var_arr

    def save_gain(self):

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        gain_file = rds.GainRef()
        gain_file['meta'] = self.meta
        gain_file['data'] = self.gain * u.electron / u.DN

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': gain_file}
        af.write_to(self.outfile)


def sort_pairs(input_data):
    """
    """

