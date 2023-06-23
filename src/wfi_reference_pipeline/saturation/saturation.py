import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
from astropy import units as u
import numpy as np


class Saturation(ReferenceFile):
    """
    Class Saturation() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_saturation() creates the asdf saturation file.
    """

    def __init__(self, saturation_image, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_saturation.asdf'

        if not bit_mask:
            bit_mask = np.zeros((4096, 4096), dtype=np.uint32)

        # Access methods of base class ReferenceFile
        super(Saturation, self).__init__(saturation_image, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with gain file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI saturation reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'SATURATION'
        else:
            pass

    def make_saturation(self):
        """
        The method make_saturation() generates a saturation asdf file.

        Parameters
        ----------

        Outputs
        -------
        af: asdf file tree: {meta, data, dq}
            meta:
            data:
            dq: mask - data quality array
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the gain object from the data model.
        saturationfile = rds.SaturationRef()
        saturationfile['meta'] = self.meta
        saturationfile['data'] = self.input_data * u.DN
        saturationfile['dq'] = self.mask
        # Saturation files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': saturationfile}
        af.write_to(self.outfile)
