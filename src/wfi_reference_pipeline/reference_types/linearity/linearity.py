import logging
import warnings
from collections.abc import Iterable

import asdf
import astropy.units as u
import numpy as np
import roman_datamodels.stnode as rds
from astropy.stats import sigma_clipped_stats
from romancal.lib import dqflags

from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_REF_OPTICAL_ELEMENT_DARK,
    WFI_REF_OPTICAL_ELEMENT_GRISM,
    WFI_REF_OPTICAL_ELEMENT_PRISM,
    WFI_REF_OPTICAL_ELEMENTS,
)

from ..reference_type import ReferenceType

# logging.getLogger('stpipe').setLevel(logging.WARNING)
# log_file_str = 'linearity_dev.log'
# logging.basicConfig(filename=log_file_str, level=logging.INFO)
# logging.info(f'Dark reference file log: {log_file_str}')

# dq flags
key_nl = 'NONLINEAR'  # Pixel is non linear
key_nlc = 'NO_LIN_CORR'  # No linear correction is available
# I assume that this value corresponds when the fit is not good anymore
# but I am not checking the residuals so this is not used so far
flag_nl = dqflags.pixel[key_nl]
# Using this one for failed fits
flag_nlc = dqflags.pixel[key_nlc]

bad_optical_elements = [WFI_REF_OPTICAL_ELEMENT_DARK, WFI_REF_OPTICAL_ELEMENT_GRISM,
                        WFI_REF_OPTICAL_ELEMENT_PRISM]


class Linearity(ReferenceType):
    """
    Class Linearity() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_linearity() creates the asdf linearity file.
    """

    def __init__(self, linearity_image, meta_data, optical_element=None,
                 bit_mask=None, outfile=None, clobber=False):
        """
        Parameters
        ----------

        linearity_image: numpy.ndarray; Input (typically a flat-field image) image used
         to perform the linearity fit. It populates self.input_data.
        meta_data: dict; Metadata that will populate the output linearity reference
         file.
        clobber: bool; If True, overwrite previously generated linearity file in outfile.
        outfile: str; Path to output linearity file. (Default: roman_linearity.asdf).
        bit_mask: numpy.ndarray; Input mask (dq array from the flat file to use).
        """
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_linearity.asdf'
        self.img_size = linearity_image.shape[1:]
        # Make sure that the shapes match if initialized to None
        if bit_mask is None:
            bit_mask = np.zeros(linearity_image.shape, dtype=np.uint32)

        # Access methods of base class ReferenceType
        super(Linearity, self).__init__(linearity_image, meta_data, bit_mask=bit_mask,
                                        clobber=clobber)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI linearity reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'LINEARITY'
        else:
            pass



        self.times = None
        # Get whether we are in spectroscopic or imaging mode
        im_mode = None
        if optical_element in WFI_REF_OPTICAL_ELEMENTS:
            if optical_element not in bad_optical_elements:
                im_mode = WFI_MODE_WIM
        if (im_mode is None) or (im_mode == 'OTHER'):
            raise ValueError('The selected optical element is DARK or not recognized')

        self.times = WFI_FRAME_TIME[im_mode] * np.arange(self.input_data.shape[0])
        self.fit_complete = False
        self.poly_order = None
        self.coeffs = None

    def make_linearity(self, poly_order=6, constrained=False):
        """
        The method make_linearity() populates the coefficients of a Linearity object,
        fitting the linearity coefficients for a single flat image.

        Parameters
        ----------
        poly_order: integer; Polynomial order to use for the fits.
        constrained: bool; If True, it returns the fit resulting fixing intercept
                     to 0 and slope to 1.
        """

        # Get the dimensions of the image
        npix_0 = self.img_size[0]
        npix_1 = self.img_size[1]

        # Load input image
        if np.isscalar(self.times):
            time = np.arange(0, self.input_data.shape[0]) * self.times
        else:
            if self.times.shape[0] != self.input_data.shape[0]:
                raise ValueError('Frame times should have the same length as datacube')

        nframes = get_fit_length(self.input_data, self.times, dq=self.dq_mask)

        # Keep only the frames that we need
        img_arr = self.input_data[:nframes, :, :]
        time = self.times[:nframes]
        if self.dq_mask is not None:
            if len(self.dq_mask.shape) == 2:
                img_dq = np.ones(nframes)[:, None, None] * self.dq_mask
            elif len(self.dq_mask.shape) == 3:
                img_dq = self.dq_mask[:nframes, :, :]
            else:
                raise ValueError('dq array expected to be 2 or 3-dimensional')

        if not constrained:
            aux_arr = np.ma.array(img_arr)
            aux_arr.mask = (img_dq != 0)
            coeffs = np.ma.polyfit(time, aux_arr.reshape(nframes, -1), poly_order)
            coeffs = coeffs.reshape(-1, npix_0, npix_1)
        elif (constrained) & (np.allclose(img_dq, np.zeros_like(img_dq))):
            # np.polyfit does not allow for fixed coefficients because
            # it is solving a linear algebra problem (that's why it's fast).
            # In order to fix coefficients we have to do some math.
            # Based on solution here:
            # https://stackoverflow.com/questions/48469889/how-to-fit-a-polynomial-with-some-of-the-coefficients-constrained
            v = np.vander(time, poly_order + 1)
            # Removing the last column of the Vandermonde matrix
            # is equivalent to setting a0 to 0 -> they go from order n to 0
            v = np.delete(v, -1, axis=1)
            # The above slicing it's a view, it creates a read-only array...
            img_arr = np.copy(img_arr.reshape(nframes, -1))
            # Subtract t from the original array, i.e., remove the linear part
            # so the fit is still correct
            img_arr -= v[:, -1, None]
            # Now drop that column from the Vandermonde matrix
            v = np.delete(v, -1, axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(v, img_arr, rcond=None)
            coeffs = coeffs.reshape(-1, npix_0, npix_1)
            # Insert the slope=1 and intercept=0
            coeffs = np.insert(coeffs, poly_order - 1,
                               np.ones((npix_0, npix_1)), axis=0)
            coeffs = np.insert(coeffs, poly_order,
                               np.zeros((npix_0, npix_1)), axis=0)
        else:
            raise NotImplementedError
        # Mask bad pixels
        self.dq_mask[np.where(np.isnan(coeffs))[1:]] += flag_nlc
        self.coeffs = coeffs
        self.fit_complete = True
        self.poly_order = poly_order

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the linearity object from the data model.
        linearity_datamodel_tree = rds.LinearityRef()
        linearity_datamodel_tree['meta'] = self.meta
        if self.fit_complete:
            nonlinear_pixels = ((self.dq_mask == float('NaN')) |
                                (self.coeffs[0, :, :] == float('NaN')))
            nonlinear_pixels = np.where(nonlinear_pixels)
            self.dq_mask[nonlinear_pixels] += flag_nlc  # linearity correction not available
            self.coeffs[self.coeffs == float('NaN')] = 0
            self.coeffs = self.coeffs.astype(np.float32)  # to comply with datamodel
            linearity_datamodel_tree['coeffs'] = self.coeffs * u.DN
            linearity_datamodel_tree['dq'] = np.sum(self.dq_mask, axis=0).astype(np.uint32)
        else:
            linearity_datamodel_tree['coeffs'] = np.zeros((1, self.img_size[0],
                                                           self.img_size[1]), np.float32)
            linearity_datamodel_tree['dq'] = np.sum(self.dq_mask, axis=0).astype(np.uint32)
        return linearity_datamodel_tree

    def save_linearity(self, datamodel_tree=None, clobber=False):
        """
        Save a linearity reference file to an asdf file.

        Parameters
        ----------
        clobber: bool; If True, it allows overwritting a previous linearity file.
        """
        self.clobber = clobber
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)


def get_fit_length(datacube, time, dq=None, frac_thr=0.5,
                   nsigma=3):
    """
    Function to obtain the frames in the datacube to use for the linearity fits.

    Parameters:
    -----------
    datacube: numpy.ndarray; Datacube with shape (Nreads, Npix, Npix)
              containing all reads of a given image.

    time: numpy.ndarray; time at which the frames were taken.

    frac_thr: float; Maximum fraction of flagged pixels to consider a read
              as ``good`` to obtain a baseline standard deviation within a read.
              Default 0.5.

    nsigma: int; Threshold to consider a pixel in the fit. If the difference
            between reads is larger than nsigma * sigma, the pixel is considered good.

    Outputs:
    --------
    nframes: int; Number of frames to consider for the fit.

    """
    if len(datacube.shape) != 3:
        raise ValueError('A 3-dimensional datacube is expected')
    if dq is not None:
        if len(dq.shape) == 2:
            dq = np.ones(datacube.shape[0])[:, None, None] * dq
        elif len(dq.shape) == 3:
            if (dq.shape != datacube.shape):
                raise ValueError('''A 3d dq array should have the same
                                    dimensions as the image datacube''')
            else:
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
            frac_bad = np.count_nonzero(dq[base_read, :, :]) / len(dq[base_read, :, :])
            # If the fraction of bad pixels is higher than the threshold, move to the next
            # read
            if frac_bad > frac_thr:
                base_read += 1
            else:
                break
        # Check if all the reads were bad, and if so, just mask everything
        if base_read == len(time):
            return 0
        good_dqs = dq[base_read, :, :] == 0
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :][good_dqs])
    else:
        _, _, std = sigma_clipped_stats(datacube[base_read, :, :])

    # If the gradient is bigger than 3-times the standard deviation we
    # are accumulating charge -- we follow NIRCam's algorithm here

    logging.debug('Standard deviation estimate:', std)
    if dq is None:
        try:
            # Get the first frame at which there are some signs of saturation in a pixel
            nframes = np.where(grad < nsigma * std)[0][0]
        except IndexError:
            # If there are no saturated frames, we use all of them
            nframes = datacube.shape[0]
    else:
        try:
            # When applying a mask, it flattens the arrray, so we need to unravel
            nframes = np.unravel_index(np.where(grad[dq == 0] < nsigma * std)[0][0],
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
    coeffs: numpy.ndarray; Output linearity coefficients.
    mask: numpy.ndarray; Output bit-mask.
    unc: numpy.ndarray; Uncertainty of the linearity coefficients.
    """
    # Linearity files do not have data quality or error arrays.
    err = None
    coeffs = None
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
                coeffs = input_lin.coeffs
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
                        npx0 = lin.coeffs.shape[1]
                        npx1 = lin.coeffs.shape[2]
                        use_lin.append(lin)
                    else:
                        if (lin.coeffs.shape[1] == npx0) & (lin.coeffs.shape[2] == npx1):
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
                    if constrained:  # The different count rates are already accounted for
                        img_arr_all += np.polyval(lin.coeffs, t_grid[:, None, None])
                    else:
                        # It the slope and intercept are not 1, and the count rates
                        # are different, we have to normalize everything to the same
                        # count rate.
                        # Evaluate polynomial
                        aux_img = np.polyval(lin.coeffs, t_grid[:, None, None])
                        # Subtract intercept to make it zero
                        aux_img -= lin.coeffs[lin.poly_order, :]
                        # Subtract linear term (and leave slope 1)
                        aux_sl = lin.coeffs[lin.poly_order - 1, :] - 1
                        aux_img -= aux_sl * t_grid[:, None, None]
                        img_arr_all += aux_img
                        if use_unc:
                            # This is to get the std as <mean^2> - <mean>^2
                            if constrained:
                                img_arr_all_sq += np.polyval(lin.coeffs,
                                                             t_grid[:, None, None])**2
                            else:
                                img_arr_all_sq += aux_img**2
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
                    # We do not want to use the weights nor want an uncertainty on the fit
                    else:
                        coeffs = np.polyfit(t_grid,
                                            img_arr_all.reshape(nframes_grid, -1),
                                            poly_order)
                    coeffs = coeffs.reshape(-1, npx0, npx1)

                # Case where we want to use the uncertainties as weights
                else:
                    wgt = 1. / var
                    # non-finite entries have zero weight
                    wgt[~np.isfinite(wgt)] = 0.
                    # Multiplying weights and measurements
                    wy = wgt * img_arr_all
                    wy = wy.reshape(nframes_grid, -1)
                    # Clear memory
                    del (img_arr_all_sq)
                    del (img_arr_all)
                    # Here is where the linear algebra fun starts...
                    # using the weighted least squares estimator
                    # https://en.wikipedia.org/wiki/Weighted_least_squares
                    v = np.vander(t_grid, poly_order + 1)
                    _aux1 = np.einsum('ij, ik, il', v, wgt.reshape(nframes_grid, -1), v)
                    _aux1 = np.linalg.inv(_aux1.swapaxes(1, 0))  # (V^T W V)^-1
                    _aux2 = np.einsum('ijk, lj', _aux1, v)  # (V^T W V)^-1 V^T
                    coeffs = np.einsum('ijk, ki -> ji', _aux2, wy)
                    # If we want to get uncertainties in the parameters
                    if return_unc:
                        err = np.sqrt(np.diagonal(_aux1, axis1=1, axis2=2))
                        err = err.T.reshape(-1, npx0, npx1)

                    coeffs = coeffs.reshape((poly_order + 1, npx0, npx1))
    else:
        raise ValueError('Linearity images not fit!')

    if output_file is not None:
        _save_linearity(output_file, meta, coeffs.astype(np.float32),
                        mask, clobber=clobber)
    else:
        # We want the option to get the arrays without writing (e.g., for testing)
        if not return_unc:
            return coeffs, mask
        else:
            return coeffs, mask, err


def _save_linearity(outfile, meta, coeffs, mask, clobber=False, unc=None):
    """
    Save a linearity reference file to an asdf file. This is now a bit redundant
    but allows make_linearity_multi to save the coefficients to a file.

    Parameters
    ----------
    outfile: str; Output path for the linearity file.
    meta: dict; Metadata of the output file.
    coeffs: numpy.ndarray; Array containing the values of the coefficient of the
     polynomial fit for the linearity reference file.
    mask: numpy.ndarray; Output bit mask.
    clobber: bool; If True, it allows overwritting a previous linearity file.
    unc: numpy.ndarray; Uncertainty of the coefficients, coeffs.
    """
    linearityfile = rds.LinearityRef()
    linearityfile['meta'] = meta
    linearityfile['coeffs'] = coeffs
    nonlinear_pixels = (mask == float('NaN')) | (coeffs[0, :, :] == float('NaN'))
    nonlinear_pixels = np.where(nonlinear_pixels)
    mask[nonlinear_pixels] += flag_nlc  # linearity correction not available
    coeffs[coeffs == float('NaN')] = 0  # Set to zero the NaN pixels
    linearityfile['dq'] = mask
    if unc is not None:
        linearityfile['unc'] = unc
    # Add in the meta data and history to the ASDF tree.
    af = asdf.AsdfFile()
    af.tree = {'roman': linearityfile}
    af.write_to(outfile)
