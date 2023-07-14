import roman_datamodels.stnode as rds
import numpy as np
from ..utilities.reference_file import ReferenceFile
import asdf


class ReferencePixel(ReferenceFile):
    """
    Class InvLinearity() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, input_data, meta_data, outfile='roman_refpix.asdf', gamma=None, zeta=None, alpha=None,
                 bit_mask=None, clobber=False, ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------

        input_data: numpy.ndarray; Placeholder. It populates self.input_data.
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = roman_inv_linearity.asdf
            Filename with path for saved inverse linearity reference file.
        gamma: 2D complex128 numpy array
        zeta: 2D complex128 numpy array
        alpha: 2D complex128 numpy array
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer array for supplying a mask for the creation of the dark reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        """

        # Access methods of base class ReferenceFile
        super(ReferencePixel, self).__init__(input_data, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=False)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI reference pixel reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'REFPIX'

        # Initialize attributes
        self.outfile = outfile
        self.gamma = gamma
        self.zeta = zeta
        self.alpha = alpha
        self.refpix_obj = None

    def make_referencepixel_coeffs(self):
        """
        The method make_inv_linearity_obj creates an object from the DMS data model.
        """

        self.gamma = np.zeros((4096, 4096), dtype=np.complex128)
        self.zeta = np.zeros((4096, 4096), dtype=np.complex128)
        self.alpha = np.zeros((4096, 4096), dtype=np.complex128)

    def make_referencepixel_obj(self):
        """
        The method make_inv_linearity_obj creates an object from the DMS data model.
        """

        # Construct the dark object from the data model.
        self.refpix_obj = rds.RefpixRef()
        self.refpix_obj['meta'] = self.meta
        self.refpix_obj['gamma'] = self.gamma
        self.refpix_obj['zeta'] = self.zeta
        self.refpix_obj['alpha'] = self.alpha

    def save_referencepixel(self):
        """
        The method save_referencepixel writes the reference file object to the specified asdf outfile.
        """

        # af: asdf file tree: {meta, gamma, zeta, alpha}
        af = asdf.AsdfFile()
        af.tree = {'roman': self.refpix_obj}
        af.write_to(self.outfile)


