import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile

from .irrc_extract_ramp_sums import extract
from .irrc_generate_weights import generate

import asdf
import os                       # Operating system
import shutil
import logging
from ..utilities import logging_functions
logging_functions.configure_logging("ReferencePixel")


class ReferencePixel(ReferenceFile):
    """
    Class ReferencePixel() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written.
    Under automated operations conditions, a list of dark calibration files from a directory will be the input data for the class to begin generatoring a reference pixel reference pixel.
    Each file will be run through IRRC, which will generate per expsoure sums necessary to minimize Fn = alpha*Fa + gamma*Fl + zeta*Fr, which are then combined to create a final reference pixel reference file coefficientts.  
    """

    def __init__(self, input_data, meta_data, outfile='roman_refpix.asdf', gamma=None, zeta=None, alpha=None, 
                 bit_mask=None, clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------

        input_data: string object; default = None
            List of dark calibration filenames with absolute paths for one detector
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = roman_refpix.asdf
            Filename with path for saved reference correction reference file.
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

    def make_referencepixel_coeffs(self, tmppath=None):
        """
        The method make_referencepixel_coeffs creates an object from the DMS data model. The method make_referencepixel_coeffs() ingests all files located in a directory as a python object list of filenames with absolute paths.  The reference pixel reference file is created by iterating through each dark calibration file and computing a model of the read noise in the normal pixels that is a linear combination of the reference output, left, and right column pixels (IRRC; Rauscher et al., in prep).  The sums for each exposure are then combined/summed together to create a final model that minimizes the 1/f noise given as
        Fn = alpha*Fa + gamma+Fl + zeta+Fr.  The coefficients are then saved as a final reference class attribute.  

        NOTE: Initial testing was performed by S. Betti with 100 dark files each with 55 reads.  Each file took ~134s to calculate the sums and ~4hours to compute the final coefficients.  The peak memory usage was 40 GB.  

        Parameters
        ----------
        tmppath: str; default = None
            Path string. Absolute or relative path for temporary storage of IRRC sums for individual ramps
            By default, None is provided and the method below produces the folder in the location the script is run.  
        """

        # we assume files is a list of exposures paths and filename for 1 detector. 
        files = self.input_data

        # Display the directory name where the dark calibration files are located to make the reference pixel reference file
        logging.info(
            f"Using files from {os.path.dirname(files[0])} to construct reference coeffienct object."
        )
        
        detector = self.meta['instrument']['detector'] # taken from WFIMetaReferencePixel() dictionary style
        logging.info(f'detector {detector}')

        # create name of folder to save the exposure weight .h5 files
        if tmppath is None:
            tmpdir = f'./tmp_IRRC_{detector}'
        else:
            tmpdir = os.path.join(tmppath, f'tmp_IRRC_{detector}')

        # if tmpdir does not exist, create it
        if not os.path.exists(tmpdir):
            logging.info(f'creating temporary folder to save individual exposure sums at : {tmpdir}')
            os.mkdir(tmpdir)

        # ===== Compute IRRC sums for individaul ramps =====
        logging.info('*** Compute IRRC sums for individual ramps...')
        for file in files:
                logging.info(f"*** Processing ramp: {file}")
                # Save the result in current location in folder temp_{detectr}. 
                extract(file, tmpdir)
        
        # ===== Compute IRRC frequency dependent weights =====
        # This uses the full data set.
        logging.info('*** Generate IRRC calibration file...')
        glob_pattern = tmpdir + '/*_sums.h5'
 
        # Generate frequency dependent weights
        alpha, gamma, zeta = generate(glob_pattern) 
        
        self.gamma = gamma 
        self.zeta = zeta 
        self.alpha = alpha 

        #TODO add in the files to the metadata as list

        # ===== Clean up =====
        # Delete intermediate results and folder
        shutil.rmtree(tmpdir)


    def make_hdf5_weights(self):
        """
        open an .h5 weight file and convert to .asdf file
        """
        # we assume files is a list of weight paths
        files = self.input_data
        if isinstance(files, str):
            files = [files]

        for file in files:
            with h5py.File(file, 'r') as hf:
                self.alpha = hf["alpha"][:]
                self.gamma = hf["gamma"][:]
                self.zeta = hf["zeta"][:]

        

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


