import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats


class Flat(ReferenceFile):

    """
    Class Flat() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_flat() removes outliers and divides data by mean value and
    then creates the flat asdf file containing the flattened frame.
    """

    def __init__(
        self,
        ramp_image,
        meta_data,
        bit_mask=None,
        outfile=None,
        clobber=False
    ):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_flat.asdf'

        if not bit_mask:
            bit_mask = np.zeros((4088, 4088), dtype=np.uint32)

        # Access methods of base class ReferenceFile
        super(Flat, self).__init__(
            ramp_image,
            meta_data,
            bit_mask=bit_mask,
            clobber=clobber)

        # Update metadata with flat file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI flat reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'FLAT'
        else:
            pass

    def average_flat_cubes(self):
        """
        The method make_flat() generates a flat asdf file where input data is divided
        by the mean value. The flattened image file is used to normalize quantum
        efficiency variations.

        Parameters
        ----------
        low_qe_threshold: float; default = 0.2,
           Limit below which to flag pixels as low quantum efficiency.
        low_qe_bit: unsigned integer; 13,
            Power for base 2 flag in dq array (2^13).

        Outputs
        -------
        af: asdf file tree: {meta, data, dq, err}
            meta:
            data: image divided by its mean
            dq: mask - data quality array
                masked loq QE pixels in image flagged 2**13
            err: zeros
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Normalize the flat_image by the sigma-clipped mean.
        mean, _, _ = sigma_clipped_stats(self.input_data)
        self.input_data /= mean

    def fit_flat_ramp(self):

        pass


    def calc_flat_error(self, low_qe_threshold=0.2):

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

        self.mask[self.dark_rate_image > self.hot_pixel_rate] += self.dqflag_defs['HOT']
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
        flat_datamodel_tree['data'] = self.input_data
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
