import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats
import warnings


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
        self.unc = None

    def make_linearity(self, input_files, poly_order=5, constrained=False,
                       nframes_grid=10, use_unc=False, return_unc=False,
                       clobber=False):
        """
        The method make_linearity() generates a linearity asdf file.

        Parameters
        ----------
        input_files: str or list of str; File or list of flat files to
                     use to compute the linearity.
        poly_order: integer; Polynomial order to use for the fits.
        constrained: bool; If True, it returns the fit resulting fixing intercept
                     to 0 and slope to 1.
        nframes_grid: integer; Number of points in the grid to evaluate the fit.
        use_unc: bool; If True, it uses the spread in the fits to compute the best-fit.
        return_unc: bool; If True, it returns the uncertainty of the fit.
        clobber: bool; If True, overwrite the previous file.

        Outputs
        -------
        af: asdf file tree: {meta, coeffs, dq}
            meta:
            coeffs:
            dq: mask
        """
        self.clobber = clobber
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Linearity files do not have data quality or error arrays.
        if input_files is not None:
            if isinstance(input_files, str):
                # If it's a string, then it's a single file
                coeffs, mask = self.fit_single(input_files, poly_order=poly_order,
                                               constrained=constrained)
                self.data = coeffs
            elif isinstance(input_files, list):
                # If it's a list, iterate
                coeff_list = []
                tmin = 1e24
                tmax = -1e24
                # Fit each image in the list and save the coefficients
                for idx, fname in enumerate(input_files):
                    aux_coeffs, mask, tt = self.fit_single(fname, poly_order=poly_order,
                                                           constrained=constrained,
                                                           return_time=True)
                    if idx == 0:
                        npx0 = aux_coeffs.shape[1]
                        npx1 = aux_coeffs.shape[2]
                    coeff_list.append(aux_coeffs)
                    if np.min(tt) < tmin:
                        tmin = np.min(tt)
                    if np.max(tt) > tmax:
                        tmax = np.max(tt)
                # Set the grid to evaluate the polynomials between the min and max time
                # TODO check if tmax should be the minimum of all the tmax's
                t_grid = np.linspace(tmin, tmax, nframes_grid)
                img_arr_all = np.zeros((nframes_grid, npx0, npx1))
                if use_unc:
                    img_arr_all_sq = np.zeros_like(img_arr_all)
                # Here we compute the mean of the polynomial evaluated in a grid
                # as suggested by S. Casertano
                for coeff in coeff_list:
                    if constrained:  # The different count rates are already taken into account
                        img_arr_all += np.polyval(coeff, t_grid[:, None, None])
                    else:
                        # It the slope and intercept are not 1, and the count rates
                        # are different, we have to normalize everything to the same
                        # count rate.
                        img_arr_all += (np.polyval(coeff, t_grid[:, None, None]) -
                                        coeff[poly_order, :]) / coeff[poly_order-1, :]
                    if use_unc:
                        # This is to get the std as <mean^2> - <mean>^2
                        if constrained:
                            img_arr_all_sq += np.polyval(coeff, t_grid[:, None, None])**2
                        else:
                            img_arr_all_sq += ((np.polyval(coeff, t_grid[:, None, None]) -
                                               coeff[poly_order, :])/coeff[poly_order-1, :])**2

                img_arr_all /= len(coeff_list)  # <X>
                # Use the spread between fits (not coefficients!) as uncertainty for the final fit
                if use_unc:
                    # Polyfit unfortunately does not like 2d weights, so
                    # we have to compute this stuff by hand
                    img_arr_all_sq /= len(coeff_list)  # <X^2>
                    del(coeff_list)
                    var = img_arr_all_sq - img_arr_all**2
                    # If there's no variance, there's no point in using weights...
                    if np.allclose(var, np.zeros_like(var)):
                        raise Warning('No variance in images, ignoring variance')
                        if return_unc:
                            coeffs, _cov = np.polyfit(t_grid,
                                                      img_arr_all.reshape(nframes_grid, -1),
                                                      poly_order,
                                                      cov=return_unc)
                            err = np.sqrt(np.diagonal(_cov, axis1=0, axis2=1))
                            self.unc = err.T
                        else:
                            coeffs = np.polyfit(t_grid,
                                                img_arr_all.reshape(nframes_grid, -1),
                                                poly_order)
                        coeffs = coeffs.reshape(-1, npx0, npx1)
                    else:  # The weights make sense
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
                        if return_unc:
                            err = np.sqrt(np.diagonal(_aux1, axis1=1, axis2=2))
                            self.unc = err.T.reshape(-1, npx0, npx1)

                    # Ignore spread and consider all steps in the ramp with the same weight
                else:
                    if return_unc:
                        coeffs, _cov = np.polyfit(t_grid,
                                                  img_arr_all.reshape(nframes_grid, -1),
                                                  poly_order,
                                                  cov=return_unc)
                        err = np.sqrt(np.diagonal(_cov, axis1=0, axis2=1))
                        self.unc = err.reshape((poly_order+1, npx0, npx1))
                    else:
                        coeffs = np.polyfit(t_grid, img_arr_all.reshape(nframes_grid, -1),
                                            poly_order)

                self.data = coeffs.reshape((poly_order+1, npx0, npx1))
                if 'input_files' not in self.meta.keys():
                    # Use this or the uri from the asdf files?
                    self.meta['input_files'] = input_files
            else:
                warnings.warn('Linearity images not fit, creating dummy files...', Warning)

        # Construct the linearity object from the data model.
        linearityfile = rds.LinearityRef()
        linearityfile['meta'] = self.meta
        linearityfile['coeffs'] = self.data.astype(np.float32)
        if self.unc is not None:
            linearityfile['unc'] = self.unc.astype(np.float32)
        else:
            linearityfile['unc'] = self.unc
        nonlinear_pixels = np.where((self.mask == float('NaN')) |
                                    (self.data[0, :, :] == float('NaN')))
        self.mask[nonlinear_pixels] = 2 ** 20  # linearity correction not available
        linearityfile['dq'] = self.mask
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': linearityfile}
        af.write_to(self.outfile)

    def fit_single(self, input_file, poly_order=5, constrained=False,
                   return_time=False):
        """
        Method to fit the linearity coefficients for a single flat image.

        Parameters:
        -----------
        input_file: str; File to use to obtain the linearity coefficients.
        poly_order: integer; Polynomial order to use for the fits.
        constrained: bool; If True, it will fix intercept to 0 and slope to 1.
        return_time: bool; If True, it will return the time array.
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
            time = np.arange(0, img_arr.shape[0])*time
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
                coeffs, _ = np.linalg.leastsq(V, img_arr.reshape(nframes, -1), rcond=None)
                coeffs = coeffs.reshape(-1, npix_0, npix_1)
                # Insert the slope=1 and intercept=0
                coeffs = np.insert(coeffs, poly_order-1,
                                   np.ones(npix_0, npix_1))
                coeffs = np.insert(coeffs, poly_order,
                                   np.zeros(npix_0, npix_1))
        else:
            if not constrained:
                aux_arr = np.ma.array(img_arr)
                aux_arr.mask = (img_dq != 0)
                coeffs = np.ma.polyfit(time, aux_arr.reshape(nframes, -1), poly_order)
                coeffs = coeffs.reshape(-1, npix_0, npix_1)
            else:
                raise NotImplementedError
        # Mask bad pixels
        mask[np.where(np.isnan(coeffs))[1:]] = 2**20

        if return_time:
            return coeffs, mask, time
        else:
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
