import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats


class Linearity(ReferenceFile):

    """
    Class Linearity() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_linearity() creates the asdf linearity file.
    """

    def __init__(self, linearity_image, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_linearity.asdf'

        # Access methods of base class ReferenceFile
        super(Linearity, self).__init__(linearity_image, meta_data, bit_mask=bit_mask,
                                        clobber=clobber)

        # Update metadata with gain file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI linearity reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'LINEARITY'
        else:
            pass

    def make_linearity(self, input_files, poly_order=5):
        """
        The method make_linearity() generates a linearity asdf file.

        Parameters
        ----------
        input_files: str or list of str; File or list of flat files to
                     use to compute the linearity.
        poly_order: integer; Polynomial order to use for the fits.

        Outputs
        -------
        af: asdf file tree: {meta, coeffs, dq}
            meta:
            coeffs:
            dq: mask
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the linearity object from the data model.
        linearityfile = rds.LinearityRef()
        linearityfile['meta'] = self.meta
        linearityfile['coeffs'] = self.data
        nonlinear_pixels = np.where(self.mask == float('NaN'))
        self.mask[nonlinear_pixels] = 2 ** 20  # linearity correction not available
        linearityfile['dq'] = self.mask
        # Linearity files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': linearityfile}
        af.write_to(self.outfile)

    def make_linearity_single(self, input_file, poly_order=5, constrained=False):
        """
        Method to compute the linearity coefficients for a single flat dark_image

        Parameters:
        -----------
        input_file: str; File to use to obtain the linearity coefficients.
        poly_order: integer; Polynomial order to use for the fits.
        constrained: bool; If True, it will fix intercept to 0 and slope to 1.

        Outputs
        -------
        poly: (numpy.ndarray); Array containing the coefficients of the best-fit
              polynomial for the given image.
        """

        # Load input image
        img_in = asdf.AsfFile.open(input_file)

        time = img_in['roman']['meta']['exposure']['frame_time']
        img_arr = img_in['roman']['data']
        if np.isscalar(time):
            time = np.ones(img_arr.shape[0])*time
        else:
            if time.shape[0] != img_arr.shape[0]:
                raise ValueError('Frame times should have the same length as datacube')
        try:
            img_dq = img_in['roman']['dq']
        except KeyError:
            img_dq = None
            raise Warning('dq array not found!')

        good_px = get_fit_mask(img_arr, time, dq=img_dq)
        if not constrained:
            continue
        else:
            continue


def get_fit_mask(datacube, time, dq=None, frac_thr=0.5,
                 nsigma=3, verbose=False):
    """
    Function to obtain the range in the datacube to use for the fits

    Parameters:
    -----------
    datacube: (numpy.ndarray); Datacube with shape (Nreads, Npix, Npix)
              containing all reads of a given image.
    time: (numpy.ndarray); time at which the frames were taken.
    frac_thr: (float); Maximum fraction of flagged pixels to consider a read
              as ``good`` to obtain a baseline standard deviation within a read.
              Default 0.5.
    nsigma: (int); Threshold to consider a pixel in the fit. If the difference
        between reads is larger than nsigma * sigma, the pixel is considered good.
    verbose: (bool); If `True` it shows debugging messages.

    Outputs:
    --------
    data_out: (numpy.ndarray); Boolean mask for the datacube with shape
              (Nreads, Npix, Npix). `True` indicates that the pixel will be used
              for the fit.
    """
    # We compute the gradient in counts
    # we just want to check that the signal grows by a quantity
    # larger than nsigma * std (i.e., it's likely not saturated)
    grad = np.gradient(datacube, axis=0)
    # Compute standard deviation in the first read as a reference
    # as long as the first read has more than 50 percent of the pixels
    # If not, we will move to a different read.
    base_read = 0
    if dq is not None:
        for i in range(len(time)):
            # Check the fraction of flagged pixels
            frac_bad = np.count_nonzero(dq[base_read, :, :])/len(dq[base_read, :, :])
            # If the fraction of bad pixels is higher than the threshold, move to the next
            # read
            if frac_bad > frac_thr:
                base_read += 1
            else:
                break
        # Check if all the reads were bad, and if so, just mask everything
        if base_read == len(time):
            return np.zeros(datacube.shape, dtype=bool)
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :][dq[base_read, :, :] == 0])
    else:
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :])

    # If the gradient is bigger than 3-times the standard deviation we
    # are accumulating charge -- we follow NIRCam's algorithm here

    if verbose:
        print('Standard deviation estimate:', std)

    good_px = (grad > 3*std)  # TODO: check if this is right with non-evenly spaced reads
    if dq is not None:
        good_px &= (dq == 0)
    return good_px
