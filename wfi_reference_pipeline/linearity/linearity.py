import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np


class Linearity(ReferenceFile):

    """
    Class Linearity() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_linearity() creates the asdf linearity file.
    """

    def __init__(self, linearity_image, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_linearity.asdf'

        # Access methods of base class ReferenceFile
        super(Linearity, self).__init__(linearity_image, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with gain file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI linearity reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'LINEARITY'
        else:
            pass

    def make_linearity(self):
        """
        The method make_linearity() generates a linearity asdf file.

        Parameters
        ----------

        Outputs
        -------
        af: asdf file tree: {meta, coeffs, dq}
            meta:
            coeffs:
            dq: mask
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the linearity object from the data model.
        linearityfile = rds.LinearityRef()
        linearityfile['meta'] = self.meta
        linearityfile['coeffs'] = self.data
        nonleainr_pixels = np.where(self.mask == float('NaN'))
        self.mask[nonleainr_pixels] += 2 ** 20 # linearity correction not available
        linearityfile['dq'] = self.mask
        # Linearity files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': linearityfile}
        af.write_to(self.outfile)
