import unittest
import numpy as np
import numpy.testing as testing
import tempfile
from astropy.time import Time
from ..linearity.linearity import Linearity, fit_single
from ..linearity.linearity import get_fit_length, make_linearity_multi
import os
import shutil


def setup_dummy_meta():
    _meta = dict()
    _meta['useafter'] = Time.now().iso
    _meta['pedigree'] = 'DUMMY'
    _meta['instrument'] = dict()
    _meta['instrument']['name'] = 'WFI'
    _meta['instrument']['detector'] = 'WFI01'
    return _meta


class GetFitLengthTestCase(unittest.TestCase):
    def test_get_fit_length_unmasked(self):
        """
        Test function that detects the length of the datacube to
        use during the fits.
        """
        t_test = 10
        poly_coeffs = (1., 0.1)
        y_test = np.polyval(poly_coeffs, t_test)
        with self.assertRaises(ValueError):
            nf = get_fit_length(y_test, t_test)

        t_test = np.array([10, 20])
        y_test = np.polyval(poly_coeffs,
                            t_test)[:, None, None]*np.ones((2, 2), dtype=np.float32)
        nf = get_fit_length(y_test, t_test)
        testing.assert_almost_equal(len(t_test), nf)

    def test_get_fit_length_masked(self):
        """
        Test the get_fit_length function where all elements are masked
        """
        poly_coeffs = (1., 0.1)
        t_test = np.array([10, 20])
        y_test = np.polyval(poly_coeffs,
                            t_test)[:, None, None]*np.ones((2, 2), dtype=np.float32)
        dq_test = np.array([0, 1])[:, None, None]*np.eye(2, dtype=np.uint32)
        nf = get_fit_length(y_test, t_test, dq=dq_test)
        testing.assert_equal(len(t_test), nf)
        dq_test[1, :, :] = 1
        nf = get_fit_length(y_test, t_test, dq=dq_test)
        # The dq array is considered for the fits but does not change the length
        testing.assert_equal(len(t_test), nf)
        dq_test = np.ones((2, 2, 2), dtype=np.uint32)
        nf = get_fit_length(y_test, t_test, dq=dq_test)
        testing.assert_equal(0, nf)


class FitSingleTestCase(unittest.TestCase):
    def test_fit_single_nodq(self):
        """
        Test fitting a single image.
        """

        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        coeffs, mask = fit_single(y_test, t_test, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 0, 0])

    def test_fit_single_dq(self):
        """
        Test fitting a single image with a dq array.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        dq_test = np.zeros(y_test.shape, dtype=np.uint32)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        coeffs, mask = fit_single(y_test, t_test, img_dq=dq_test,
                                  poly_order=2)
        # Check the 0,0 pixel that has some masked values
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 0, 0])
        # Check some unmasked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 1, 1])
        # Check the other masked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10])

    def test_fit_single_nodq_wnoise(self):
        """
        Test fitting a single image with no dq, but noise added to it.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        y_test += 0.005*np.random.normal(size=y_test.shape)
        coeffs, mask = fit_single(y_test, t_test, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)

    def test_fit_single_dq_wnoise(self):
        """
        Test fitting a single image with dq and noise.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        y_test += 0.005*np.random.normal(size=y_test.shape)
        dq_test = np.zeros(y_test.shape, dtype=np.uint32)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        coeffs, mask = fit_single(y_test, t_test, img_dq=dq_test,
                                  poly_order=2)
        # Check the 0,0 pixel that has some masked values
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 0, 0], decimal=2)
        # Check some unmasked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 1, 1], decimal=2)
        # Check the other masked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)

    def test_fit_single_wrong_poly(self):
        """
        Test fitting a single image requesting a polynomial order larger
        than the original.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        # Check if the "correct" polynomial is still fit even if we change
        # input and output polynomial orders don't match
        coeffs, mask = fit_single(y_test, t_test, img_dq=dq_test,
                                  poly_order=5)
        # Check the 0,0 pixel that has some masked values
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 0, 0])
        # Check some unmasked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 1, 1])
        # Check the other masked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 10, 10])

    def test_fit_single_constrained_dq(self):
        """
        Test fitting a single image with constraints and a dq array.
        It should raise a NotImplementedError.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 1, 0)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20

        # We have not implemented this option with a dq array yet
        with self.assertRaises(NotImplementedError):
            fit_single(y_test, t_test, img_dq=dq_test,
                       poly_order=2, constrained=True)

    def test_fit_single_constrained_nodq(self):
        """
        Test fitting a single image fixing slope to 1 and intercept to 0.
        """
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 1, 0)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        # Check if the "correct" polynomial is still fit even if we change
        # input and output polynomial orders don't match
        coeffs, mask = fit_single(y_test, t_test, poly_order=2, constrained=True)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10])


class LinearityTestCase(unittest.TestCase):
    def test_make_linearity_dummy(self):
        """
        Test the method make_linearity() if no input is
        passed,, and a dummy reference file is created
        """
        # Prepare the metadata that needs to be passed to Linearity
        _meta = setup_dummy_meta()

        dummy_img = np.zeros((1, 1, 1), np.float32)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(dummy_img, _meta, outfile=outfile)
        with testing.assert_warns(Warning):
            lin.make_linearity(dummy_img)

        # Check that the file is created
        assert(os.path.exists(lin.outfile))
        lin = Linearity(dummy_img, _meta, outfile=outfile)
        # Check that the file is not overwritten
        with self.assertRaises(FileExistsError):
            lin.make_linearity(dummy_img)

    def test_make_linearity_nodq(self):
        """
        Test fit to single linearity file
        """

        # Set up a dummy file
        _meta = setup_dummy_meta()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        # Run without noise
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile)
        lin.mask = None
        lin.make_linearity(poly_order=2)
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 0, 0])

    def test_make_linearity_dq(self):
        _meta = setup_dummy_meta()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, bit_mask=dq_test, outfile=outfile)
        lin.make_linearity(poly_order=2)
        # Check the 0,0 pixel that has some masked values
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 0, 0])
        # Check some unmasked pixel
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 1, 1])
        # Check the other masked pixel
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 10, 10])

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


class LinearityMultiTestCase(unittest.TestCase):
    def test_make_linearity_multi_dummy(self):
        _meta = setup_dummy_meta()
        t_test = np.linspace(0, 10, 20)
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        with testing.assert_warns(Warning):
            _ = make_linearity_multi(None, _meta, output_file=None)

    def test_make_linearity_multi_dq_nounc(self):
        _meta = setup_dummy_meta()
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 10, 0.01)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        noise1 = 0.005*np.random.normal(size=y_test.shape)
        noise2 = 0.005*np.random.normal(size=y_test.shape)
        noise3 = 0.005*np.random.normal(size=y_test.shape)
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin1 = Linearity(y_test+noise1, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        lin2 = Linearity(y_test+noise2, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        lin3 = Linearity(y_test+noise3, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        with self.assertRaises(ValueError):
            make_linearity_multi(lin1, lin1.meta)
        with self.assertRaises(ValueError):
            make_linearity_multi([lin1, lin2], lin1.meta)
        lin1.make_linearity(poly_order=2)
        lin2.make_linearity(poly_order=2, clobber=True)
        lin3.make_linearity(poly_order=2, clobber=True)
        # Check that it works with one file
        with testing.assert_warns(Warning):
            coeffs, mask = make_linearity_multi(lin1, lin1.meta, poly_order=2, output_file=None)
            testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)
        # Check with a list
        coeffs, mask = make_linearity_multi([lin1], lin1.meta, poly_order=2, output_file=None)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)

        # Check that it works with 3 objects (with 2 or more objects we homogeinize the data
        # to slope 1 and intercept 0 to alllow for different bias levels and count rates)
        coeffs, mask = make_linearity_multi([lin1, lin2, lin3], lin1.meta,
                                            poly_order=2, output_file=None)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)
        # Check that it works with the wrong poly order
        coeffs, mask = make_linearity_multi([lin1, lin2], lin1.meta, poly_order=5, output_file=None)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[3:, 10, 10], decimal=2)
        # Check returning uncertainty
        with testing.assert_warns(Warning):
            coeffs, mask, err = make_linearity_multi(lin1, lin1.meta, poly_order=2, return_unc=True,
                                                     output_file=None)
            testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)

        coeffs, mask, err = make_linearity_multi([lin1], lin1.meta, poly_order=2,
                                                 output_file=None, return_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)
        coeffs, mask, err = make_linearity_multi([lin1, lin2], lin1.meta, poly_order=2,
                                                 return_unc=True, output_file=None)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)

    def test_make_linearity_multi_dq_useunc(self):
        _meta = setup_dummy_meta()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 10, 0.01)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        noise1 = 0.005*np.random.normal(size=y_test.shape)
        noise2 = 0.005*np.random.normal(size=y_test.shape)
        noise3 = 0.005*np.random.normal(size=y_test.shape)
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin1 = Linearity(y_test+noise1, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        lin2 = Linearity(y_test+noise2, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        lin3 = Linearity(y_test+noise3, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32), clobber=True)
        # Check that it works with one file
        lin1.make_linearity(poly_order=2)
        lin2.make_linearity(poly_order=2, clobber=True)
        lin3.make_linearity(poly_order=2, clobber=True)
        with testing.assert_warns(Warning):
            coeffs, mask = make_linearity_multi(lin1, lin1.meta, poly_order=2, output_file=None,
                                                use_unc=True)
            testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)
        # Check with a list
        with testing.assert_warns(Warning):
            # With a single element the variance is zero
            coeffs, mask = make_linearity_multi([lin1], lin1.meta, poly_order=2, output_file=None,
                                                use_unc=True)
            testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                        coeffs[:, 10, 10], decimal=2)

        # Check that it works with 3 objects (with 2 or more objects we homogeinize the data
        # to slope 1 and intercept 0 to alllow for different bias levels and count rates)
        coeffs, mask = make_linearity_multi([lin1, lin2, lin3], lin1.meta,
                                            poly_order=2, output_file=None, use_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)
        # Check that it works with the wrong poly order
        coeffs, mask = make_linearity_multi([lin1, lin2], lin1.meta, poly_order=5, output_file=None,
                                            use_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[3:, 10, 10], decimal=2)
        # Check returning uncertainty
        with testing.assert_warns(Warning):
            coeffs, mask, err = make_linearity_multi(lin1, lin1.meta, poly_order=2, return_unc=True,
                                                     output_file=None, use_unc=True)
            testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)
        coeffs, mask, err = make_linearity_multi([lin1, lin2], lin1.meta, poly_order=2,
                                                 return_unc=True, output_file=None, use_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    coeffs[:, 10, 10], decimal=2)

    def test_make_linearity_dq_ngrid(self):
        _meta = setup_dummy_meta()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 10, 0.01)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        noise1 = 0.005*np.random.normal(size=y_test.shape)
        noise2 = 0.005*np.random.normal(size=y_test.shape)
        dq_test = np.zeros_like(y_test)
        # Flag some pixels
        dq_test[0, 0, 0] = 2**20
        dq_test[14, 0, 0] = 2**20
        dq_test[15, 0, 0] = 2**20
        dq_test[14, 10, 10] = 2**20
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin1 = Linearity(y_test+noise1, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32))
        lin2 = Linearity(y_test+noise2, _meta, outfile=outfile,
                         bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                           dtype=np.uint32))
        lin1.make_linearity(poly_order=2)
        # Check that it breaks with one file
        with self.assertRaises(ValueError):
            make_linearity_multi(lin1, lin1.meta, poly_order=4, nframes_grid=3)
        with self.assertRaises(ValueError):
            make_linearity_multi([lin1], lin1.meta, poly_order=4, nframes_grid=3,
                                 clobber=True)
            with self.assertRaises(ValueError):
                make_linearity_multi([lin1, lin2], lin1.meta, poly_order=4, nframes_grid=3,
                                     clobber=True)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
