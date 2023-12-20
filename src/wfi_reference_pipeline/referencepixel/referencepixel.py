import roman_datamodels.stnode as rds
import numpy as np
from ..utilities.reference_file import ReferenceFile
import asdf

from irrc_extract_ramp_sums import *
from irrc_generate_weights import *

class ReferencePixel(ReferenceFile):
    """
    Class InvLinearity() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, input_data, meta_data, outfile='roman_refpix.asdf', freq = None, gamma=None, zeta=None, alpha=None,
                 bit_mask=None, clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------

        input_data: numpy.ndarray; Placeholder. It populates self.input_data.
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = roman_refpix.asdf
            Filename with path for saved reference correction reference file.
        freq: 
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
        super().__init__(input_data, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=False)

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

    def make_referencepixel_coeffs(self):
        """
        The method make_referencepixel_coeffs creates an object from the DMS data model.
        """


        print('*** Compute IRRC sums for individual ramps...')
        for file in files:
            name = file.split('/')[-1] + '_sums.h5' # SKB: don't redo ones that have been done 
            if name in caldat: # SKB: don't redo ones that have been done 
                print('###########', file, 'completed!!  ###########')
                print()
                print()
                print()
            else:
                print("*** Processing ramp: ", file)
                # Save the result in /tmp. On ADAPT, they have it set up so that each
                # user has a subdirectory in /tmp. The format is /tmp/mpl_<user_name>.
                # extract(datdir + '/' + file, tmpdir)
                extract(file, tmpdir)
                print()
                print()
                print()
        
        # ===== Compute IRRC frequency dependent weights =====
        # This uses the full data set.
        print('*** Generate IRRC calibration file...')
        # The way that Steve has set this up, the first argument is really just a glob
        # pattern. It is not an actual list of files.
        glob_pattern = tmpdir + '/' + '*_' + detector + '_*_sums.h5'

        # Make the output filename
        calfil = libdir + '/irrc_weights_' + detector + '_' + date_beg + '.h5'
 
        # Generate frequency dependent weights
        generate(glob_pattern, calfil)
        
        # ===== Clean up =====
        # Delete intermediate results
        files = glob(glob_pattern)
        for file in files:
            os.remove(file)


        # self.gamma = np.zeros((32, 286721), dtype=np.complex128)
        # self.zeta = np.zeros((32, 286721), dtype=np.complex128)
        # self.alpha = np.zeros((32, 286721), dtype=np.complex128)

        # print('testing initalize_coeffs_array()')
        # print(self.gamma.real)

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        referencepixel_datamodel_tree = rds.RefpixRef()
        referencepixel_datamodel_tree['meta'] = self.meta
        referencepixel_datamodel_tree['gamma'] = self.gamma
        referencepixel_datamodel_tree['zeta'] = self.zeta
        referencepixel_datamodel_tree['alpha'] = self.alpha

        print('testing populate_datamodel_tree')
        print(self.gamma.real)

        return referencepixel_datamodel_tree

    def save_referencepixel(self, datamodel_tree=None):
        """
        The method save_referencepixel writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)


