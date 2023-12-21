import roman_datamodels.stnode as rds
import numpy as np
from ..utilities.reference_file import ReferenceFile
import asdf
import logging
# logging = logging.getLogger('ReferencePixel')
from ..utilities import logging_functions
logging_functions.configure_logging("ReferencePixel")


import os                       # Operating system
import shutil
import h5py                     # Needed to read the calibration file that we make > HAD TO INSTALL

from .irrc_extract_ramp_sums import *
from .irrc_generate_weights import *

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
            ASSUME IT IS A LIST OF FILES!!
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
        self.freq = freq

    def make_referencepixel_coeffs(self, tmppath=None):
        """
        The method make_referencepixel_coeffs creates an object from the DMS data model.

        Parameters
        ----------
        tmppath: str; default = None
            Path string. Absolute or relative path for temporary storage of IRRC sums for individual ramps
            By default, None is provided and the method below produces the folder in the location the script is run.  
        """

        # we assume files is a list of exposures paths and filename for 1 detector. 
        files = self.input_data

        # assume detector SCA is in self.meta_data()
        detector = self.meta['instrument']['detector'] # taken from WFIMetaReferencePixel() dictionary style
        logging.info(f'detector {detector}')

        # create name of folder to save the exposure weight .h5 files
        if tmppath is None:
            tmpdir = f'./tmpIRRC_{detector}'
        else:
            tmpdir = os.path.join(tmppath, f'tmpIRRC_{detector}')

        # if tmpdir does not exist, create it
        if not os.path.exists(tmpdir):
            logging.info(f'creating folder: {tmpdir}')
            os.mkdir(tmpdir)

        logging.info('*** Compute IRRC sums for individual ramps...')
        for file in files:
                logging.info(f"*** Processing ramp: {file}")
                # Save the result in current location in folder temp_{detectr}. 
                extract(file, tmpdir)
        
        # ===== Compute IRRC frequency dependent weights =====
        # This uses the full data set.
        logging.info('*** Generate IRRC calibration file...')
        # The way that Steve has set this up, the first argument is really just a glob
        # pattern. It is not an actual list of files.
        glob_pattern = tmpdir + '/*_sums.h5'
 
        # Generate frequency dependent weights
        freq, alpha, gamma, zeta = generate(glob_pattern) 
        
        self.gamma = gamma 
        self.zeta = zeta 
        self.alpha = alpha 
        self.freq = freq 

        # ===== Clean up =====
        # Delete intermediate results and folder
        shutil.rmtree(tmpdir)



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
        referencepixel_datamodel_tree['freq'] = self.freq

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


