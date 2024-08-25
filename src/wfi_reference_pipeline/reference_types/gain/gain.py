import logging
import numpy as np
import roman_datamodels.stnode as rds
from astropy import units as u
from wfi_reference_pipeline.resources.wfi_meta_saturation import WFIMetaGain


# @log_info
class Gain(ReferenceType):
    """
    Class Gain() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_gain() creates the asdf gain file.
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_gain.asdf",
            clobber=False,
    ):

        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_flat.asdf
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
        if not isinstance(meta_data, WFIMetaGain):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaGain"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI gain reference file."

        logging.debug(f"Default gain reference file object: {outfile} ")

        # Attributes to make reference file with valid data model.
        self.gain_image = None  # The attribute 'data' in data model.
        self.num_files = 0

    def make_gain_image(self):

        self.gain_image =

    def calculate_mean_variance(self):
        """
        The method make_gain() uses the photon transfer curve to estimate
        the gain in units of electrons/DN.

        Parameters
        ----------

        Outputs
        -------
        None
        """

        data_shape = self.data[0].shape
        n_pairs = len(self.data) // 2

        # Set up the variance and mean signal level arrays that
        # we'll need for the photon transfer curve. For a full WFI
        # SCA, these should be N x 4088 x 4088, where N is the number
        # of reads or resultants in the input data.
        var_arr = np.zeros(data_shape, dtype=np.float)
        signal_arr = np.zeros(data_shape, dtype=np.float)

        # For each read/resultant, step over each pair of flats and
        # compute the difference. Save the difference to a temporary
        # N x 4088 x 4088 array, where N is the number of pairs.
        #
        # From the temporary array, compute the variance and store it
        # in the plane of var_arr corresponding to a given read/resultant.
        for result in range(data_shape[0]):
            temp_arr = np.zeros((n_pairs, data_shape[1], data_shape[2]),
                                dtype=np.float)
            for p in range(n_pairs):
                temp_arr[p] = self.data[2 * p][result] - \
                              self.data[(2 * p) + 1][result]

            # Compute the variance.
            mean_diff = np.mean(temp_arr, axis=0)
            numerator = np.sum((temp_arr - mean_diff) ** 2, axis=0)
            var_arr[result] += (numerator / (2 * n_pairs))

        # Compute the mean signal per resultant.
        signal_arr = np.mean(self.data, axis=0)

        return signal_arr, var_arr

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        gain_datamodel_tree = rds.GainRef()
        gain_datamodel_tree['meta'] = self.meta
        gain_datamodel_tree['data'] = self.gain_image * u.electron / u.DN

        return gain_datamodel_tree
