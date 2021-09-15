import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf


class Gain(ReferenceFile):
    """ The parent class Gain() inherits the ReferenceFile child class methods
    where static meta data for all file types are written. The method
    make_gain() generates the asdf file matching the dimensions of the input data array.
    """

    def __init__(self, gain_image, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_gain.asdf'

        # Access methods of base class ReferenceFile
        super(Gain, self).__init__(gain_image, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with gain file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Gain file default description.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype']     = 'GAIN'
        else:
            pass

    def make_gain(self):
        """ The method make_gain() generates a gain reference file type
        that matches the dimensions of the input data array.
        """

        # Construct the gain object from the data model.
        gainfile         = rds.GainRef()
        gainfile['meta'] = self.meta
        gainfile['data'] = self.data
        # Gain files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af      = asdf.AsdfFile()
        af.tree = {'roman': gainfile}
        af.write_to(self.outfile)
