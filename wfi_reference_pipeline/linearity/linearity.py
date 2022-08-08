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

    def fit_single(self, input_file, poly_order=5, constrained=False):
        """
        Method to fit the linearity coefficients for a single flat image.

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

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Load input image
        img_in = asdf.AsdfFile.open(input_file)

        time = img_in['roman']['meta']['exposure']['frame_time']
        img_arr = img_in['roman']['data']
        if np.isscalar(time):
            time = np.ones(img_arr.shape[0])*time
        else:
            if time.shape[0] != img_arr.shape[0]:
                raise ValueError('Frame times should have the same length as datacube')
        if 'dq' in img_in['roman'].keys():
            img_dq = img_in['roman']['dq']
        else:
            img_dq = None

        nframes = get_fit_length(img_arr, time, dq=img_dq)

        # Keep only the frames that we need
        img_arr = img_arr[:nframes, :, :]
        time = time[:nframes]
        if img_dq is not None:
            img_dq = img_dq[:nframes, :, :]
        npix_0 = img_arr.shape[1]
        npix_1 = img_arr.shape[2]
        mask = np.zeros((npix_0, npix_1))
        if (img_dq is None):
            if not constrained:
                coeffs = np.polyfit(time, img_arr.reshape(nframes, -1), poly_order)
                coeffs = coeffs.reshape(-1, npix_0, npix_1)
            else:
                # np.polyfit does not allow for fixed coefficients because
                # it is solving a linear algebra problem (that's why it's fast).
                # In order to fix coefficients we have to do some math.
                # Based on solution here:
                # https://stackoverflow.com/questions/48469889/how-to-fit-a-polynomial-with-some-of-the-coefficients-constrained
                V = np.vander(time, poly_order+1)
                # Removing the last column of the Vandermonde matrix
                # is equivalent to setting a0 to 0 -> they go from order n to 0
                V = np.delete(V, -1, axis=1)
                # Subtract t from the original array, i.e., remove the linear part
                # so the fit is still correct
                img_arr -= V[:, -1, None]
                # Now drop that column from the Vandermonde matrix
                V = np.delete(V, -1, axis=1)
                coeffs, _ = np.linalg.leastsq(V, img_arr.reshape(nframes, -1))
                coeffs = coeffs.reshape(-1, npix_0, npix_1)
                # Insert the slope=1 and intercept=0
                coeffs = np.insert(coeffs, poly_order-1,
                                   np.ones(npix_0, npix_1))
                coeffs = np.insert(coeffs, poly_order,
                                   np.zeros(npix_0, npix_1))
        else:
            # For loop over all pixels -- slow but safe
            coeffs = np.zeros((poly_order+1, npix_0, npix_1))
            for i in range(npix_0*npix_1):
                ipx, jpx = np.unravel_index(i, (npix_0, npix_1))
                dq_here = img_dq[:, ipx, jpx]
                img_here = img_arr[:, ipx, jpx]
                if np.count_nonzero(dq_here != 0) > poly_order:
                    # There are more parameters than points to fit
                    coeffs[:, ipx, jpx] = np.nan
                    mask[ipx, jpx] = 2**20
                else:
                    # Fit the relevant frames
                    frames_good = np.where(dq_here == 0)[0]
                    aux_coeff = np.polyfit(frames_good, img_here[frames_good], deg=poly_order)
                    coeffs[:, ipx, jpx] = aux_coeff
        return coeffs, mask


def get_fit_length(datacube, time, dq=None, frac_thr=0.5,
                   nsigma=3, verbose=False):
    """
    Function to obtain the frames in the datacube to use for the linearity fits.

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
    nframes: (int); Number of frames to consider for the fit.
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
    if dq is None:
        # Get the first frame at which there are some signs of saturation in a pixel
        nframes = np.where(grad < nsigma*std)[0][0]  # TODO: check this with non-evenly spaced reads
    else:
        # When applying a mask, it flattens the arrray, so we need to unravel the indices
        nframes = np.unravel_index(np.where(grad[dq == 0] < nsigma*std)[0][0], datacube.shape)[0]
    return nframes
