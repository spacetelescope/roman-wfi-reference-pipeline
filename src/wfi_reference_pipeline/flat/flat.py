import logging

import asdf
import numpy as np
import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
from wfi_reference_pipeline.constants import WFI_FRAME_TIME, WFI_MODE_WIM
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
        input_flat_cube=None
    ):

        # Input dimensions of science array for ReferenceFile() to
        # to properly generate dq array mask for Flat().
        if bit_mask is None:
            bit_mask = np.zeros((4088, 4088), dtype=np.uint32)

        # Access methods of base class ReferenceFile
        super().__init__(
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
        self.input_read_cube = input_flat_cube  # Supplied input read cube.
        # Internal
        self.rate_array = None  # 2D rate array of the science pixels.
        self.rate_var_array = None  # 3D cute of rate arrays.
        # Flattened products
        self.flat_rate_image = None  # The attribute assigned to the flat['data'].
        self.flat_rate_image_var = None  # Variance in fitted rate image.
        self.flat_intercept = None  # Intercept image from ramp fit.
        self.flat_intercept_var = None  # Variance in fitted intercept image.
        self.flat_error = None  # The attribute assigned to the flat['err'].

        # Input data property attributes: must be a square cube of dimensions n_reads x ni x ni.
        self.ni = 4088
        self.frame_time = None  # Frame time from ancillary data.
        self.time_arr = None  # Time array of an exposure.

    def make_flat_rate_image(self):
        """
        This method determines the flow of the module based on the input data
        when the class is instantiated. The flat rate image is created at the
        end of the method to be the data in the datamodel.
        """

        # Use input files if they exist.
        if self.input_data is not None:
            print("Got files so making flat rate image from filelist")
            rate_image = self.make_flat_from_files()
        # Use input cube if it exists.

        try:
            shape = self.input_read_cube.shape
            if len(shape) == 2:
                print("Cube is already a 2D image to flatten")
                rate_image = self.input_read_cube
            elif len(shape) == 3:
                n_reads = shape[0]
                rate_image, rate_image_var = self.make_flat_from_cube(n_reads)
                print("Flat image created from cube with", n_reads, "reads")
        except AttributeError:
            print("Input cube does not have a shape attribute or is not a numpy array.")

        # Normalize the flat_image by the sigma-clipped mean.
        mean, _, _ = sigma_clipped_stats(rate_image)
        self.flat_rate_image = rate_image / mean
        self.calc_flat_error()

    def make_flat_from_files(self):
        """
        Go through the files supplied to the module and generate a
        cube of rate images into an array. This method uses the
        make_flat_from_cube method also so that the number of reads
        can vary over multiple input read cubes. The average of all
        rate arrays in the cube are averaged and returned.
        return:

        Returns
        -------
        avg_rate_image: 2D array;
            The average of the rate_image_array in the z axis.
        """

        print("Inside make_flat_from_files() method.")
        n_files = len(self.input_data)
        n_reads_per_fl_arr = np.zeros(n_files)
        rate_image_array = np.zeros((n_files, self.ni, self.ni), dtype=np.float32)
        rate_image_var_array = np.zeros((n_files, self.ni, self.ni), dtype=np.float32)
        for fl in range(0, n_files):
            tmp = asdf.open(self.input_data[fl], validate_on_read=False)
            n_reads_per_fl_arr[fl], _, _ = np.shape(tmp.tree["roman"]["data"])
            self.input_read_cube = tmp.tree["roman"]["data"]
            rate_image, rate_image_var = self.make_flat_from_cube(n_reads=n_reads_per_fl_arr[fl])
            rate_image_array[fl, :, :] = rate_image
            rate_image_var_array[fl, :, :] = rate_image_var
            tmp.close()

        avg_rate_image = np.mean(rate_image_array, axis=0)
        return avg_rate_image

    def make_flat_from_cube(self, n_reads, ni=None):
        """
        Method finds the fitted rate and variance by first initialize arrays
        by the number of reads and the number of pixels. The fitted ramp or slope
        along the time axis for the number of reads in the cube using a 1st order
        polyfit. The best fit solutions and variance are returned.

        Parameters
        ----------
        n_reads: integer; Positional required.
            Number of reads to initialize fitted arrays.
        ni: integer; Default: None.
            Number of reads to initialize.

        Returns
        -------
        rate_image: 2D array;
            The fitted rate image from the cube.
        rate_image_var: 2D array;
            The variance of the fitted rate image from the cube.
        """

        # If ni is supplied, overwrite attribute.
        if ni is not None:
            self.ni = ni

        # Make the time array for the length of the dark read cube exposure.
        if self.meta['instrument']['optical_element']:
            self.frame_time = WFI_FRAME_TIME[WFI_MODE_WIM]  # frame time in imaging mode in seconds
        else:
            raise ValueError('Optical element not found; this might not be a flat file.')
        time_array = np.array(
            [self.frame_time * i for i in range(1, n_reads + 1)]
        )

        p, c = np.polyfit(time_array,
                          self.input_read_cube.reshape(len(time_array), -1), 1, full=False, cov=True)

        # Reshape results back to 2D arrays.
        rate_image = p[0].reshape(self.ni, self.ni).astype(np.float32)  # the fitted ramp slope image
        rate_image_var = c[0, 0, :].reshape(self.ni, self.ni).astype(np.float32)  # covariance matrix slope variance

        return rate_image, rate_image_var

    def calc_flat_error(self):

        # TODO for future implementation
        # high_flux_err = 1.2 * self.flat_rate_image * (n_reads * 2 + 1) /
        # (n_reads * (n_reads * 2 - 1) * self.frame_time)

        #
        self.flat_error = np.random.randint(1, 11, size=(4088, 4088)).astype(np.float32) / 100.

    def update_dq_mask(self, low_qe_threshold=0.2):
        """
        Update data quality array bit mask with flag integer value.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        """

        # TODO remove random loq qe pixels from flat_rate_image
        # Generate between 200-300 pixels with low qe for DMS builds
        rand_num_lowqe = np.random.randint(200, 300)
        coords_x = np.random.randint(0, 4088, rand_num_lowqe)
        coords_y = np.random.randint(0, 4088, rand_num_lowqe)
        rand_low_qe_values = np.random.randint(5, 20, rand_num_lowqe) / 100.  # low eq in range 0.05 - 0.2
        self.flat_rate_image[coords_x, coords_y] = rand_low_qe_values

        self.low_qe_threshold = low_qe_threshold

        logging.info('Flagging low quantum efficiency pixels and updating DQ array.')
        # Locate low qe pixel ni,nj positions in 2D array
        self.mask[self.flat_rate_image < self.low_qe_threshold] += self.dqflag_defs['LOW_QE']

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the flat field object from the data model.
        flat_datamodel_tree = rds.FlatRef()
        flat_datamodel_tree['meta'] = self.meta
        flat_datamodel_tree['data'] = self.flat_rate_image
        flat_datamodel_tree['dq'] = self.mask
        flat_datamodel_tree['err'] = self.flat_error

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
