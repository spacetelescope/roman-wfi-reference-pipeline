import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats
import warnings
from collections.abc import Iterable


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
        if 'exposure' in self.meta.keys():
            if 'frame_time' in self.meta['exposure'].keys():
                self.times = self.meta['exposure']['frame_time']  # For now using this
        else:
            self.times = None
        self.fit_complete = False
        self.poly_order = None

    def make_linearity(self, poly_order=6, constrained=False):
        """
        The method make_linearity() populates the data of a Linearity object.

        Parameters
        ----------
        poly_order: integer; Polynomial order to use for the fits.
        constrained: bool; If True, it returns the fit resulting fixing intercept
                     to 0 and slope to 1.
        nframes_grid: integer; Number of points in the grid to evaluate the fit.

        Outputs
        -------
        af: asdf file tree: {meta, coeffs, dq}
            meta:
            coeffs:
            dq: mask
        """

        if (self.data is not None) & (self.times is not None):
            self.data, self.mask = fit_single(self.data, self.times, img_dq=self.mask,
                                              poly_order=poly_order, constrained=constrained)
            # Mark that the fit is complete
            self.fit_complete = True
            self.poly_order = poly_order
        else:
            raise ValueError('Input data is require to create linearity file')

    def save_linearity(self, clobber=False):
        """
        Save a linearity reference file to an asdf file.

        Parameters
        ----------
        clobber: bool; If True, it allows overwritting a previous linearity file.
        """
        self.clobber = clobber
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)
        # Construct the linearity object from the data model.
        linearityfile = rds.LinearityRef()
        linearityfile['meta'] = self.meta
        linearityfile['coeffs'] = self.data
        nonlinear_pixels = np.where((self.mask == float('NaN')) |
                                    (self.data[0, :, :] == float('NaN')))
        self.mask[nonlinear_pixels] += 2 ** 20  # linearity correction not available
        self.data[self.data == float('NaN')] = 0  # Set to zero the NaN pixels
        linearityfile['dq'] = self.mask
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': linearityfile}
        af.write_to(self.outfile)


def fit_single(img_arr, time, img_dq=None, poly_order=6, constrained=False):
    """
    Method to fit the linearity coefficients for a single flat image.

    Parameters:
    -----------
    img_arr: numpy.ndarray; Array containing the image datacube to fit with shape
        (Nreads, Npix0, Npix1).
    time: float or numpy.ndarray; If a single value is supplied, time is interpreted
        to be the space between frames, if an array is supplied, it is
        interpreted to be the time at which read is performed.
    img_dq: numpy.ndarray; Array containing the data-quality flags.
    poly_order: integer; Polynomial order to use for the fits.
    constrained: bool; If True, it will fix intercept to 0 and slope to 1.

    Outputs
    -------
    poly: (numpy.ndarray); Array containing the coefficients of the best-fit
          polynomial for the given image.
    """

    # Get the dimensions of the image
    npix_0 = img_arr.shape[1]
    npix_1 = img_arr.shape[2]

    # Load input image
    if np.isscalar(time):
        time = np.arange(0, img_arr.shape[0])*time
    else:
        if time.shape[0] != img_arr.shape[0]:
            raise ValueError('Frame times should have the same length as datacube')

    nframes = get_fit_length(img_arr, time, dq=img_dq)

    # Keep only the frames that we need
    img_arr = img_arr[:nframes, :, :]
    time = time[:nframes]
    if img_dq is not None:
        if len(img_dq.shape) == 2:
            img_dq = np.ones(nframes)[:, None, None]*img_dq
        elif len(img_dq.shape) == 3:
            img_dq = img_dq[:nframes, :, :]
        else:
            raise ValueError('dq array expected to be 2 or 3-dimensional')
    mask = np.zeros((npix_0, npix_1), np.uint32)

    if (img_dq is None) or (np.allclose(img_dq, np.zeros_like(img_dq))):
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
            # The above slicing it's a view, it creates a read-only array...
            img_arr = np.copy(img_arr.reshape(nframes, -1))
            # Subtract t from the original array, i.e., remove the linear part
            # so the fit is still correct
            img_arr -= V[:, -1, None]
            # Now drop that column from the Vandermonde matrix
            V = np.delete(V, -1, axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(V, img_arr, rcond=None)
            coeffs = coeffs.reshape(-1, npix_0, npix_1)
            # Insert the slope=1 and intercept=0
            coeffs = np.insert(coeffs, poly_order-1,
                               np.ones((npix_0, npix_1)), axis=0)
            coeffs = np.insert(coeffs, poly_order,
                               np.zeros((npix_0, npix_1)), axis=0)
    else:
        if not constrained:
            aux_arr = np.ma.array(img_arr)
            aux_arr.mask = (img_dq != 0)
            coeffs = np.ma.polyfit(time, aux_arr.reshape(nframes, -1), poly_order)
            coeffs = coeffs.reshape(-1, npix_0, npix_1)
        else:
            raise NotImplementedError
    # Mask bad pixels
    mask[np.where(np.isnan(coeffs))[1:]] += 2**20

    return coeffs.astype(np.float32), mask


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
    if len(datacube.shape) != 3:
        raise ValueError('A 3-dimensional datacube is expected')
    if dq is not None:
        if len(dq.shape) == 2:
            dq = np.ones(datacube.shape[0])[:, None, None]*dq
        elif len(dq.shape) == 3:
            pass
        else:
            raise ValueError('dq array expected to be 2 or 3-dimensional')
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
            return 0
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :][dq[base_read, :, :] == 0])
    else:
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :])

    # If the gradient is bigger than 3-times the standard deviation we
    # are accumulating charge -- we follow NIRCam's algorithm here

    if verbose:
        print('Standard deviation estimate:', std)
    if dq is None:
        try:
            # Get the first frame at which there are some signs of saturation in a pixel
            nframes = np.where(grad < nsigma*std)[0][0]  # TODO: check this with not even spacing
        except IndexError:
            # If there are no saturated frames, we use all of them
            nframes = datacube.shape[0]
    else:
        try:
            # When applying a mask, it flattens the arrray, so we need to unravel the indices
            nframes = np.unravel_index(np.where(grad[dq == 0] < nsigma*std)[0][0],
                                       datacube.shape)[0]
        except IndexError:
            # If there are no saturated frames, we use all of them
            nframes = datacube.shape[0]
    return nframes


def make_linearity_multi(input_lin, meta, poly_order=6, constrained=False,
                         nframes_grid=10, use_unc=False, return_unc=False,
                         clobber=False, output_file='roman_linearity.asdf'):
    """
    Average several linearity fits and return final linearity image.

    Parameters
    ----------
    input_lin: Linearity object or iterable containing Linearity objects.
    meta: metadata; Metadata to include to the output file.
    poly_order: integer; Polynomial order to use for the fits.
    constrained: bool; If True, it returns the fit resulting fixing intercept
                 to 0 and slope to 1.
    nframes_grid: integer; Number of points in the grid to evaluate the fit.
    use_unc: bool; If True, it uses the spread in the fits to compute the best-fit.
    return_unc: bool; If True, it returns the uncertainty of the fit.
    clobber: bool; If True, overwrite the previous file.
    output_file: str; Path of output file.

    Outputs
    -------
    af: asdf file tree: {meta, coeffs, dq}
        meta:
        coeffs:
        dq: mask
    """
    # Linearity files do not have data quality or error arrays.
    err = None
    data = None
    mask = None
    if nframes_grid < poly_order:
        raise ValueError('''nframes_grid cannot be smaller than the required polynomial
                          order, as the result will be unconstrained.''')
    if input_lin is not None:
        if isinstance(input_lin, Linearity):
            warnings.warn('''Using a single linearity file,
                          ignoring grid''', Warning)
            # If it's a string, then it's a single file
            if input_lin.fit_complete:
                data = input_lin.data
            else:
                raise ValueError('''The input Linearity files have not been fit.
                                 Please run input_lin.make_linearity before running
                                 make_linearity_multi''')
        elif isinstance(input_lin, Iterable):
            # If it's a list, iterate
            tmin = 1e24
            tmax = -1e24
            # Fit each image in the list and save the coefficients
            use_lin = []
            for lin in input_lin:
                if lin.fit_complete:
                    tt = lin.times
                    if len(use_lin) < 1:
                        npx0 = lin.data.shape[1]
                        npx1 = lin.data.shape[2]
                        use_lin.append(lin)
                    else:
                        if (lin.data.shape[1] == npx0) & (lin.data.shape[2] == npx1):
                            use_lin.append(lin)
                    if np.min(tt) < tmin:
                        tmin = np.min(tt)
                    if np.max(tt) > tmax:
                        tmax = np.max(tt)
            # Check if any fits have survived
            if len(use_lin) < 1:
                raise ValueError('''At least one Linearity object with a fit completed
                                 is needed.''')
            else:  # We have one or more fits
                # Set the grid to evaluate the polynomials between the min and max time
                t_grid = np.linspace(tmin, tmax, nframes_grid)
                img_arr_all = np.zeros((nframes_grid, npx0, npx1))
                if use_unc:
                    img_arr_all_sq = np.zeros_like(img_arr_all)
                # Here we compute the mean of the polynomial evaluated in a grid
                # as suggested by S. Casertano
                for lin in use_lin:
                    if constrained:  # The different count rates are already taken into account
                        img_arr_all += np.polyval(lin.data, t_grid[:, None, None])
                    else:
                        # It the slope and intercept are not 1, and the count rates
                        # are different, we have to normalize everything to the same
                        # count rate.
                        img_arr_all += (np.polyval(lin.data, t_grid[:, None, None]) -
                                        lin.data[lin.poly_order, :]) / lin.data[lin.poly_order-1, :]
                        if use_unc:
                            # This is to get the std as <mean^2> - <mean>^2
                            if constrained:
                                img_arr_all_sq += np.polyval(lin.data, t_grid[:, None, None])**2
                            else:
                                img_arr_all_sq += ((np.polyval(lin.data, t_grid[:, None, None]) -
                                                    lin.data[lin.poly_order, :]) /
                                                   lin.data[lin.poly_order-1, :])**2
                # Now we have the sum of the polynomials in the image
                img_arr_all /= len(use_lin)  # <X>

                # And the sum of the squares if we need it
                if use_unc:
                    img_arr_all_sq /= len(use_lin)  # <X^2>
                    var = img_arr_all_sq - img_arr_all**2
                    # If there's no variance, there's no point in using weights...
                    if np.allclose(var, np.zeros_like(var)):
                        warnings.warn('No variance in images, ignoring variance', Warning)
                        use_unc = False

                # Time to run the fits

                # We start with the case where we don't use the uncertainties
                if not use_unc:
                    if return_unc:
                        coeffs, _cov = np.polyfit(t_grid,
                                                  img_arr_all.reshape(nframes_grid, -1),
                                                  poly_order,
                                                  cov=return_unc)
                        err = np.sqrt(np.diagonal(_cov, axis1=0, axis2=1)).T
                    # We do not want to use the weights nor want an uncertainty on the fits
                    else:
                        coeffs = np.polyfit(t_grid,
                                            img_arr_all.reshape(nframes_grid, -1),
                                            poly_order)
                    data = coeffs.reshape(-1, npx0, npx1)

                # Case where we want to use the uncertainties as weights
                else:
                    wgt = 1./var
                    wgt[~np.isfinite(wgt)] = 0.  # Just set to zero the weight of NaN and infs
                    # Multiplying weights and measurements
                    wy = wgt*img_arr_all
                    wy = wy.reshape(nframes_grid, -1)
                    # Clear memory
                    del(img_arr_all_sq)
                    del(img_arr_all)
                    # Here is where the linear algebra fun starts...
                    # using the weighted least squares estimator
                    # https://en.wikipedia.org/wiki/Weighted_least_squares
                    V = np.vander(t_grid, poly_order+1)
                    _aux1 = np.einsum('ij, ik, il', V, wgt.reshape(nframes_grid, -1), V)
                    _aux1 = np.linalg.inv(_aux1.swapaxes(1, 0))  # (V^T W V)^-1
                    _aux2 = np.einsum('ijk, lj', _aux1, V)  # (V^T W V)^-1 V^T
                    coeffs = np.einsum('ijk, ki -> ji', _aux2, wy)
                    # If we want to get uncertainties in the parameters
                    if return_unc:
                        err = np.sqrt(np.diagonal(_aux1, axis1=1, axis2=2))
                        err = err.T.reshape(-1, npx0, npx1)

                    data = coeffs.reshape((poly_order+1, npx0, npx1))
    else:
        raise ValueError('Linearity images not fit!')

    # Construct the linearity object from the data model.
    linearityfile = rds.LinearityRef()
    linearityfile['meta'] = meta
    if data is not None:
        linearityfile['coeffs'] = data.astype(np.float32)
    if err is not None:
        linearityfile['unc'] = err.astype(np.float32)
    if data is not None:
        # The "bad" pixels will be those where the fits did not work
        nonlinear_pixels = np.where(data[0, :, :] == float('NaN'))
        mask = np.zeros((data.shape[1], data.shape[2]), dtype=np.uint32)
        mask[nonlinear_pixels] = 2 ** 20  # linearity correction not available
        linearityfile['dq'] = mask
    if output_file is not None:
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': linearityfile}
        af.write_to(output_file)
    else:
        # We want the option to get the arrays without writing (e.g., for testing)
        if not return_unc:
            return data, mask
        else:
            return data, mask, err
