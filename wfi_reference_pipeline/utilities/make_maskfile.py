import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf


class MaskFile(ReferenceFile):
    """ The parent class MaskFile() inherits the ReferenceFile child class methods
    where static meta data for all file types are written. The method
    make_maskfile() generates the asdf file matching the dimensions of the input data array.
    """

    def __init__(self, mask_image, meta_data, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_maskfile.asdf'

        # Access methods of base class ReferenceFile
        super(MaskFile, self).__init__(mask_image, meta_data, clobber=clobber)

        # Update metadata with mask file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Mask file default description.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'MASK'
        else:
            pass

    def make_maskfile(self):
        """ The method make_maskfile() generates a mask reference file type
        that matches the dimensions of the input data array.
        """
        # Construct the read noise object from the data model.
        maskfile         = rds.MaskRef()
        maskfile['meta'] = self.meta
        maskfile['dq']   = self.mask
        # Mask files do not have data or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af      = asdf.AsdfFile()
        af.tree = {'roman': maskfile}
        af.write_to(self.outfile)
