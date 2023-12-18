import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
from wfi_reference_pipeline.constants import WFI_TYPE_IMAGE, WFI_FRAME_TIME, WFI_MODE_WIM, WFI_MODE_WSM
from astropy.stats import sigma_clipped_stats


class Flat(ReferenceFile):

    """
    Class Flat() inherits the ReferenceFile() base class methods where
    static meta data for all reference file types are written. The class
    ingests a list of files and finds all exposures with the same filter
    within some maximum date range. Fit ramps to all available filter
    cubes ro generate flat rate images and average together and normalize
    to produce the filter dependent flat rate image.
    """

    def __init__(
        self,
        flat_file_list,
        meta_data,
        bit_mask=None,
        outfile="roman_flat.asdf",
        clobber=False,
        input_read_cube=None,
        input_rate_array=None
    ):

        # Input dimensions of science array for ReferenceFile() to
        # to properly generate dq array mask for Flat().
        if bit_mask is None:
            bit_mask = np.zeros((4088, 4088), dtype=np.uint32)

        # Access methods of base class ReferenceFile
        super(Flat, self).__init__(
            flat_file_list,
            meta_data,
            bit_mask=bit_mask,
            clobber=clobber,
            make_mask=True,
        )

        # Update metadata with file type info if not included.
        if "description" not in self.meta.keys():
            self.meta["description"] = "Roman WFI flat reference file."
        if "reftype" not in self.meta.keys():
            self.meta["reftype"] = "FLAT"

        # Initialize attributes
        self.outfile = outfile
        # Additional object attributes
        # Inputs
        self.input_read_cube = input_read_cube  # Supplied input read cube.
        self.input_rate_array = input_rate_array  # Rate image from ramp fit.
        # Internal
        self.rate_array = None
        self.rate_var_array = None
        # Flattened products
        self.flat_rate_image = None
        self.flat_rate_image_var = None  # Variance in fitted rate image.
        self.flat_intercept = None  # Intercept image from ramp fit.
        self.flat_intercept_var = None  # Variance in fitted intercept image.

        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.n_files = None
        self.n_reads_per_fl = None
        self.ni = 4088
        self.frame_time = None  # Frame time from ancillary data.
        self.time_arr = None  # Time array of an exposure.

    def make_flat_rate_image(self):
        """


        """
        if self.input_data is not None:
            rate_image = self.make_flat_from_files()
        if self.input_read_cube is not None:
            n_reads, _, _ = np.shape(self.input_read_cube)
            rate_image = self.make_flat_from_cube(n_reads)
        if self.input_rate_array is not None:
            rate_image = self.input_rate_array

        # Normalize the flat_image by the sigma-clipped mean.
        mean, _, _ = sigma_clipped_stats(rate_image)
        self.flat_rate_image = rate_image / mean

    def make_flat_from_files(self):

        self.n_files = len(self.input_data)
        n_reads_per_fl_arr = np.zeros(self.n_files)
        rate_image_array = np.zeros((self.n_files, self.ni, self.ni), dtype=np.float32)
        rate_image_var_array = np.zeros((self.n_files, self.ni, self.ni), dtype=np.float32)
        for fl in range(0, self.n_files):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_reads_per_fl_arr[fl], _, _ = np.shape(tmp.tree["roman"]["data"])
            self.input_read_cube = tmp.tree["roman"]["data"]
            rate_image, rate_image_var = self.make_flat_from_cube(n_reads=n_reads_per_fl_arr[fl])
            rate_image_array[fl, :, :] = rate_image
            rate_image_var_array[fl, :, :] = rate_image_var
            tmp.close()

        avg_rate_image = np.mean(rate_image_array, axis=0)
        return avg_rate_image

    def make_flat_from_cube(self, n_reads):

        rate_image, rate_image_var, time_arr = self.initialize_arrays(n_reads=n_reads)
        rate_image, rate_image_var = self.fit_flat_ramp(time_array=time_arr)

        return rate_image, rate_image_var

    def initialize_arrays(self, n_reads, ni=None):
        """
        Method initialize_arrays makes arrays for the flat rate image of ni x xni.
        Parameters
        ----------
        ni: integer; Default=None
            Number of square pixels of array ni x ni.
        """

        rate_image = np.zeros((self.ni, self.ni), dtype=np.float32)
        rate_image_var = np.zeros((self.ni, self.ni), dtype=np.float32)

        # Make the time array for the length of the dark read cube exposure.
        if self.meta['exposure']['type'] == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            raise ValueError('Got frame time other than imaging mode; WFI Flat() is for WIM only.')
        time_arr = np.array(
            [self.frame_time * i for i in range(1, n_reads + 1)]
        )

        return rate_image, rate_image_var, time_arr

    def fit_flat_ramp(self, time_array):
        """
        The fit_flat_ramp() method computes the fitted ramp or slope along the time axis for the number of reads
        in the cube using a 1st order polyfit. The best fit solutions and variance are saved into
        attributes.
        """

        logging.info('Computing dark rate image.')
        # Perform linear regression to fit ma table resultants in time; reshape cube for vectorized efficiency.

        p, c = np.polyfit(time_array,
                          self.input_read_cube.reshape(len(time_array, -1), 1, full=False, cov=True))

        # Reshape results back to 2D arrays.
        rate_image = p[0].reshape(self.ni, self.ni).astype(np.float32)  # the fitted ramp slope image
        rate_image_var = c[0, 0, :].reshape(self.ni, self.ni).astype(np.float32)  # covariance matrix slope variance
        # If needed the flat intercept image and variance are p[1] and c[1,1,:]
        return rate_image, rate_image_var

    def calc_flat_error(self, n_reads):

        high_flux_err = 1.2 * self.flat_rate_image * (n_reads * 2 + 1) / (n_reads * (n_reads * 2 - 1) * self.frame_time)

        pass

    def update_dq_mask(self, low_qe_threshold=0.2, low_qe_bit=13):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        low_qe_bit: integer; default = 13
            DQ loq quantum efficiency pixel flag value in romancal library.
        """

        self.mask[self.flat_rate > self.hot_pixel_rate] += self.dqflag_defs['HOT']
        self.mask[(self.warm_pixel_rate <= self.dark_rate_image) & (self.dark_rate_image < self.hot_pixel_rate)] \
            += self.dqflag_defs['WARM']
        self.mask[self.dark_rate_image < self.dead_pixel_rate] += self.dqflag_defs['DEAD']

        # Generate between 200-300 pixels with low qe
        rand_num_lowqe = np.random.randint(200, 300)
        coords_x = np.random.randint(0, 4088, rand_num_lowqe)
        coords_y = np.random.randint(0, 4088, rand_num_lowqe)
        rand_low_qe_values = np.random.randint(5, 20, rand_num_lowqe) / 100. # low eq in range 0.05 - 0.2
        self.input_data[coords_x, coords_y] = rand_low_qe_values

        # Add DQ flag for low QE pixels.
        low_qe_pixels = np.where(self.input_data < low_qe_threshold)
        self.mask[low_qe_pixels] += 2 ** low_qe_bit

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the flat field object from the data model.
        flat_datamodel_tree = rds.FlatRef()
        flat_datamodel_tree['meta'] = self.meta
        flat_datamodel_tree['data'] = self.flat_rate_image
        flat_datamodel_tree['dq'] = self.mask
        flat_datamodel_tree['err'] = np.random.randint(1, 11, size=(4088, 4088)).astype(np.float32) / 100.

        return flat_datamodel_tree

    def save_flat(self, datamodel_tree=None):
        """
        The method save_flat writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
