import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import roman_datamodels as rdm
import roman_datamodels.stnode as rds
from astropy import units as u

from wfi_reference_pipeline.resources.wfi_meta_referencepixel import (
    WFIMetaReferencePixel,
)

from ..reference_type import ReferenceType
from .irrc_constants import NUM_COLS, NUM_ROWS
from .irrc_extract_ramp_sums import extract
from .irrc_generate_weights import generate

logging = logging.getLogger('ReferencePixel')

class ReferencePixel(ReferenceType):
    """
    Class ReferencePixel() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    Under automated operations conditions, a list of calibration files from a directory will be the input data for the class to begin generatoring a reference pixel reference pixel.
    Each file will be run through IRRC, which will generate per exposure sums necessary to minimize Fn = alpha*Fa + gamma*Fl + zeta*Fr, which are then combined to create a final reference pixel reference file coefficientts.

    rfp_referencepixel = RefType(meta_data, file_list=file_list)
    rfp_referencepixel.make_referencepixel_image(tmppath='/path/to/save/tmp/files/')
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
            # place in list if only one file
            if isinstance(self.file_list, str):
                self.file_list = [self.file_list]
            # Display the directory name where the calibration files are located to make the reference pixel reference file
            logging.info(f"Using files from {os.path.dirname(self.file_list[0])} to construct reference coeffienct object.")
        
        # if data array is provided instead of file list
        else:
            # confirm data type as either numpy array or quantity
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

            # check dimensions of data.  should be either 3D or 4D array. 
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
            
            
    def get_data_cube_from_file(self, file_name, skip_first_frame=False):
        """
        The method get_data_cube_from_file() opens an individual data file and extracts the data and 33rd amplifier. The files are opened using roman_datamodels and the data and amp33 arrays are extracted and concatenated into one array.   
        
        Parameters
        ----------
        file_name: str
            Path string. Full path to data data file
        skip_first_frame: boolean; default = False
            should the first frame of the data be skipped? In the majority of cases, it should not be skipped as long as the reset read is a separate frame. 
        
        Returns
        ----------
        dset.astype(np.float64): np.ndarray
            The concatenated data and amp33 exposure as one dataset with type np.float64
        """
        if not isinstance(file_name, Path):
            file_name = Path(file_name)
   
        # confirm file exisits
        if not file_name.exists():
            mesg = f'Input file {str(file_name)} does not exist. Terminating.'
            logging.fatal(mesg)
            raise FileNotFoundError(mesg)
        
        # open file using roman_datamodels
        logging.info("OPENING - " + str(file_name))
        if '.asdf' not in file_name.name:
            raise ValueError('can only read in .asdf format')
        fil = rdm.open(file_name)

        # extract data
        data = fil['data']
        amp33 = fil['amp33']
        if isinstance(data, u.Quantity):
            data = data.value
        if isinstance(amp33, u.Quantity):
            amp33 = amp33.value

        # concatentate data and amp33 together with amp33 added to end of array. 
        dset = np.concatenate([data, amp33], axis=2)
        if skip_first_frame:
            dset = dset[1:]

        # check data shape
        data_shape = dset.shape
        num_cols = data_shape[2]
        num_rows = data_shape[1]
        if num_cols != NUM_COLS or num_rows != NUM_ROWS:
            raise Exception("File:", file_name, " has incorrect dimensions.  Expecting num_rows =", NUM_ROWS, ", num_cols =", NUM_COLS)
        
        # Convert from uint16 to prepare for in-place computations
        return dset.astype(np.float64)

    def _get_detector_name_from_data_file_meta(self, file_name):
        """
        The method _get_detector_name_from_data_file_meta() opens an individual data file, extracts the instrument detector name, and updates the meta_data.instrument_detector variable. 
        
        Parameters
        ----------
        file_name: str
            Path string. Full path to data file
        """
        if not isinstance(file_name, Path):
            file_name = Path(file_name)
 
        if not file_name.exists():
            mesg = f'Input file {str(file_name)} does not exist. Terminating.'
            logging.fatal(mesg)
            raise FileNotFoundError(mesg)
        
        if '.asdf' not in file_name.name:
            raise ValueError('can only read in .asdf format')
        fil = rdm.open(file_name)
        
        # get detector
        detector = fil['meta']['instrument']['detector']
        # update detector name in meta
        self.meta_data.instrument_detector = detector

    def make_referencepixel_image(self, tmppath=None, detector_name=None, skip_first_frame=False):
        """
        The method make_referencepixel_coeffs creates an object from the DMS data model. The method make_referencepixel_coeffs() ingests all files located in a directory as a python object list of filenames with absolute paths.  The reference pixel reference file is created by iterating through each data calibration file and computing a model of the read noise in the normal pixels that is a linear combination of the reference output, left, and right column pixels (IRRC; Rauscher et al., in prep).  The sums for each exposure are then combined/summed together to create a final model that minimizes the 1/f noise given as
        Fn = alpha*Fa + gamma+Fl + zeta+Fr.  The coefficients are then saved as a final reference class attribute.

        NOTE: Initial testing was performed by S. Betti with 100 Total Noise files each with 55 reads.  Each file took ~180s to calculate the sums and ~5hours to compute the final coefficients.  The peak memory usage was 40 GB.

        Parameters
        ----------
        tmppath: str; default = None
            Path string. Absolute or relative path for temporary storage of IRRC sums for individual ramps
            By default, None is provided and the method below produces the folder in the location the script is run.
        detector_name: str; default = None
            Name of the detector in the form "WFI01". 
            If None and a file_list is provided, the detector_name will be pulled from the file meta data.  If ref_type_data is provided, a detector_name must be supplied. 
        skip_first_frame: boolean; default = False
            should the first frame of the data be skipped? (should not be skipped if the reset read is a separate frame)
        """
        # determine detector name to populate meta_data.instrument_detector

        if self.file_list:
            self._get_detector_name_from_data_file_meta(self.file_list[0])
        else:
            if detector_name is not None:
                self.meta_data.instrument_detector = detector_name
            else:
                msg = 'detector_name must be provided when ref_type_data is used'
                logging.fatal(msg)
                raise ValueError(msg) 
        logging.info(f'detector {self.meta_data.instrument_detector}')

        # create name of folder to save the exposure weight .h5 files
        if tmppath is None:
            tmpdir = Path(f'./tmp_IRRC_{self.meta_data.instrument_detector}')
        else:
            tmpdir = Path(tmppath, f'tmp_IRRC_{self.meta_data.instrument_detector}')

        # if tmpdir does not exist, create it
        if not tmpdir.exists():
            logging.info(f'creating temporary folder to save individual exposure sums at : {str(tmpdir)}')
            tmpdir.mkdir()

        # Compute IRRC sums for individaul ramps 
        logging.info('*** Compute IRRC sums for individual ramps...')
        
        # loop through each file and extract IRRC sums
        if self.file_list:
            for file in self.file_list:
                # extract data from cube
                data = self.get_data_cube_from_file(file, skip_first_frame=skip_first_frame)
                out_file_name = Path(tmpdir, os.path.basename(file) + '_sums.h5')
             
                logging.info(f"*** Processing ramp for file: {file}")
                extract(data, out_file_name)
        # loop through each data array and extract IRRC sums
        else:
            for exp, data in enumerate(self.ref_type_data):

                if skip_first_frame:
                    data = data[1:]
                logging.info(f"*** Processing ramp for exposure: {exp+1}")
                # create temporary out_file_name based on timestamp 
                timestr = time.strftime("%Y%m%d-%H%M%S")
                out_file_name = Path(tmpdir, f'exposure{exp+1}_{timestr}_sums.h5')

                extract(data, out_file_name)

        # Compute IRRC frequency dependent weights 
        logging.info('*** Generate IRRC calibration file...')
        glob_pattern = str(tmpdir) + '/*_sums.h5'

        # Generate frequency dependent weights
        alpha, gamma, zeta = generate(glob_pattern)

        # set as attributes
        self.gamma = gamma
        self.zeta = zeta
        self.alpha = alpha

        # Clean up by deleting intermediate results and folder
        shutil.rmtree(tmpdir)

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the refpix object from the data model.
        referencepixel_datamodel_tree = rds.RefpixRef()
        referencepixel_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        referencepixel_datamodel_tree['gamma'] = self.gamma
        referencepixel_datamodel_tree['zeta'] = self.zeta
        referencepixel_datamodel_tree['alpha'] = self.alpha

        return referencepixel_datamodel_tree

    def calculate_error(self):
        pass

    def update_data_quality_array(self):
        pass

