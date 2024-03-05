import roman_datamodels.stnode as rds
from ..reference_type import ReferenceType
import asdf
from astropy import units as u
import numpy as np


class Saturation(ReferenceType):
    """
    Class Saturation() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, input_data, meta_data, outfile='roman_saturation.asdf', bit_mask=None, clobber=False,
                 saturation_threshold=55000.):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        base class.

        Parameters
        ----------
        input_data: numpy.ndarray; Placeholder. It populates self.input_data.
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = roman_inv_linearity.asdf
            Filename with path for saved inverse linearity reference file.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer array for supplying a mask for the creation of the dark reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        saturation_threshold: integer; default = 55000.
            Minimum count pixel count level to be flagged saturated.
        """

        # Access methods of base class ReferenceType
        super().__init__(input_data, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=True)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI saturation reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'SATURATION'

        # Initialize attributes
        self.outfile = outfile

        # Make a uniform 2D array of the deteremined saturated threshold level.
        self.saturation_threshold = saturation_threshold
        self.saturation_array = saturation_threshold * np.ones((4096, 4096), dtype=np.float32)

    def update_dq_mask(self, saturation_bit=1):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        saturation_bit: integer; default = 1
            DQ saturated pixel flag value in romancal library.
        """

        saturated_pixels = np.where(self.saturation_array >= self.saturation_threshold)

        # Set mask DQ flag values
        self.mask[saturated_pixels] += 2 ** saturation_bit

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        saturation_datamodel_tree = rds.SaturationRef()
        saturation_datamodel_tree['meta'] = self.meta
        saturation_datamodel_tree['data'] = self.saturation_array * u.DN
        saturation_datamodel_tree['dq'] = self.mask

        return saturation_datamodel_tree

    def save_saturation(self, datamodel_tree=None):
        """
        The method save_saturation writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
