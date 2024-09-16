import logging
import asdf
import os                       # Operating system
import shutil
import numpy as np
import time
from astropy import units as u

import roman_datamodels as rdm
import roman_datamodels.stnode as rds
from wfi_reference_pipeline.resources.wfi_meta_referencepixel import WFIMetaReferencePixel
from ..reference_type import ReferenceType

from .irrc_extract_ramp_sums import extract
from .irrc_generate_weights import generate
from .irrc_constants import NUM_COLS, NUM_ROWS

logging = logging.getLogger('ReferencePixel')

class ReferencePixel(ReferenceType):
    """
    Class ReferencePixel() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    Under automated operations conditions, a list of dark calibration files from a directory will be the input data for the class to begin generatoring a reference pixel reference pixel.
    Each file will be run through IRRC, which will generate per expsoure sums necessary to minimize Fn = alpha*Fa + gamma*Fl + zeta*Fr, which are then combined to create a final reference pixel reference file coefficientts.

    rfp_referencepixel = RefType(meta_data, ref_type_data=)
    rfp_referencepixel.make_referencepixel_image()
    rfp_referencepixel.generate_outfile()
    """

    def __init__(
            self, 
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile='roman_referencepixel.asdf', 
            clobber=False,
            ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType() file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_mask.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
            )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaReferencePixel):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaReferencePixel"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI referencepixel reference file."

        logging.debug(f"Default referencepixel reference file object: {outfile} ")

        # Initialize attributes
        self.gamma = None
        self.zeta = None
        self.alpha = None

        # Module flow creating reference file
        # This can be 1+ files in list
        if self.file_list:
            if isinstance(self.file_list, str):
                self.file_list = [self.file_list]
            # Display the directory name where the dark calibration files are located to make the reference pixel reference file
            logging.info(f"Using files from {os.path.dirname(self.file_list[0])} to construct reference coeffienct object.")
        
        else:
            if not isinstance(ref_type_data,
                              (np.ndarray, u.Quantity)):
                raise TypeError(
                    "Input data is neither a numpy array nor a Quantity object."
                )
            if isinstance(ref_type_data, u.Quantity):  # Only access data from quantity object.
                self.ref_type_data = ref_type_data.value
                logging.info("Quantity object detected. Extracted data values.")
            else:
                self.ref_type_data = ref_type_data

            dim = self.ref_type_data.shape
            num_cols = dim[-1]
            num_rows = dim[-2]
            if len(dim) == 3 and num_cols == NUM_COLS and num_rows == NUM_ROWS:
                logging.info("User supplied one exposure to make reference pixel reference file.")
                self.ref_type_data = np.array([self.ref_type_data])
            elif len(dim) == 4 and num_cols == NUM_COLS and num_rows == NUM_ROWS:
                logging.info(f"User supplied {self.ref_type_data.shape[0]} exposures to make reference pixel reference file.")
            else:
                raise ValueError(
                    f"Input data is not a valid numpy array of dimension 3 or a set of 3D exposures (dim=4) OR does not have {NUM_COLS, NUM_ROWS} shape. Is amplifier 33 included in the data?"
                )
            
            
    def get_data_cube_from_dark_file(self, file_name, skip_first_frame=False):
        if not os.path.exists(file_name):
            mesg = f'Input file {file_name} does not exist. Terminating.'
            logging.fatal(mesg)
            raise FileNotFoundError(mesg)
        
        logging.info("OPENING - " + file_name)
        if '.asdf' not in file_name:
            raise ValueError('can only read in .asdf format')
        fil = rdm.open(file_name)
        # get data
        data = fil['data']
        amp33 = fil['amp33']
        if isinstance(data, u.Quantity):
            data = data.value
        if isinstance(amp33, u.Quantity):
            amp33 = amp33.value

        # combine back together > amp33 added to end of array. 
        dset = np.concatenate([data, amp33], axis=2)
        if skip_first_frame:
            dset = dset[1:]

        data_shape = dset.shape
        num_cols = data_shape[2]
        num_rows = data_shape[1]
        if num_cols != NUM_COLS or num_rows != NUM_ROWS:
            raise Exception("File:", file_name, " has incorrect dimensions.  Expecting num_rows =", NUM_ROWS, ", num_cols =", NUM_COLS)
        
        # Convert from uint16 to prepare for in-place computations
        return dset.astype(np.float64)


    def make_referencepixel_image(self, tmppath=None, skip_first_frame=False):
        """
        The method make_referencepixel_coeffs creates an object from the DMS data model. The method make_referencepixel_coeffs() ingests all files located in a directory as a python object list of filenames with absolute paths.  The reference pixel reference file is created by iterating through each dark calibration file and computing a model of the read noise in the normal pixels that is a linear combination of the reference output, left, and right column pixels (IRRC; Rauscher et al., in prep).  The sums for each exposure are then combined/summed together to create a final model that minimizes the 1/f noise given as
        Fn = alpha*Fa + gamma+Fl + zeta+Fr.  The coefficients are then saved as a final reference class attribute.

        NOTE: Initial testing was performed by S. Betti with 100 dark files each with 55 reads.  Each file took ~134s to calculate the sums and ~4hours to compute the final coefficients.  The peak memory usage was 40 GB.

        Parameters
        ----------
        tmppath: str; default = None
            Path string. Absolute or relative path for temporary storage of IRRC sums for individual ramps
            By default, None is provided and the method below produces the folder in the location the script is run.
        skip_first_frame: should the first frame of the data be skipped? (should not be skipped if the reset read is a separate frame)
        """
     
        detector = self.meta_data.instrument_detector # taken from WFIMetaReferencePixel() dictionary style
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
        
        if self.file_list:
            for file in self.file_list:
                data = self.get_data_cube_from_dark_file(file, skip_first_frame=skip_first_frame)
                out_file_name = os.path.join(tmpdir, os.path.basename(file) + '_sums.h5')
                logging.info(f"*** Processing ramp for file: {file}")
                extract(data, out_file_name)
        else:
            for exp, data in enumerate(self.ref_type_data):
                if skip_first_frame:
                    data = data[1:]
                logging.info(f"*** Processing ramp for exposure: {exp+1}")
                timestr = time.strftime("%Y%m%d-%H%M%S")
                out_file_name = os.path.join(tmpdir, f'exposure{exp+1}_{timestr}_sums.h5')
                # Save the result in current location in folder temp_{detectr}.
                extract(data, out_file_name)

        # ===== Compute IRRC frequency dependent weights =====
        # This uses the full data set.
        logging.info('*** Generate IRRC calibration file...')
        glob_pattern = tmpdir + '/*_sums.h5'

        # Generate frequency dependent weights
        alpha, gamma, zeta = generate(glob_pattern)

        self.gamma = gamma
        self.zeta = zeta
        self.alpha = alpha

        # ===== Clean up =====
        # Delete intermediate results and folder
        shutil.rmtree(tmpdir)

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        referencepixel_datamodel_tree = rds.RefpixRef()
        referencepixel_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
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

    def calculate_error(self, error_array=None):
        pass

    def update_data_quality_array(self):
        pass

