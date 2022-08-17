import unittest
import numpy as np
import numpy.testing as testing
import tempfile
from astropy.time import Time
from ..linearity.linearity import Linearity, fit_single
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


class FitSingleTestCase(unittest.TestCase):
    def test_fit_single_nodq(self):
        """
        Test fit a single image
        """

        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        coeffs, mask = fit_single(y_test, t_test, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 0, 0])

    def test_make_linearity_single_dq(self):
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

    def test_make_linearity_single_nodq_wnoise(self):
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        y_test += 0.005*np.random.normal(size=y_test.shape)
        coeffs, mask = fit_single(y_test, t_test, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10], decimal=2)

    def test_make_linearity_single_dq_wnoise(self):
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

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


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

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

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


'''
    def test_make_linearity_single_wrong_poly(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
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
        af['roman'] = dict()
        af['roman']['data'] = y_test
        af['roman']['meta'] = _meta
        af['roman']['dq'] = dq_test
        dummy_path = os.path.join(self.test_dir, 'dummy_img.asdf')
        af.write_to(dummy_path)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile)
        # Check if the "correct" polynomial is still fit even if we change
        # input and output polynomial orders don't match
        coeffs, mask = lin.fit_single(dummy_path, poly_order=5)
        # Check the 0,0 pixel that has some masked values
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 0, 0])
        # Check some unmasked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 1, 1])
        # Check the other masked pixel
        testing.assert_almost_equal(poly_coeffs, coeffs[3:, 10, 10])

    def test_make_linearity_single_constrained_dq(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
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
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        af['roman'] = dict()
        af['roman']['data'] = y_test
        af['roman']['meta'] = _meta
        af['roman']['dq'] = dq_test
        dummy_path = os.path.join(self.test_dir, 'dummy_img.asdf')
        af.write_to(dummy_path)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile)
        # We have not implemented this option with a dq array yet
        with self.assertRaises(NotImplementedError):
            lin.fit_single(dummy_path, poly_order=2, constrained=True)

    def test_make_linearity_single_constrained_nodq(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up constrained fit (i.e, slope 1 and intercept 0)
        poly_coeffs = (0.1, 1, 0)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        af['roman'] = dict()
        af['roman']['data'] = y_test
        af['roman']['meta'] = _meta
        dummy_path = os.path.join(self.test_dir, 'dummy_img.asdf')
        af.write_to(dummy_path)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile)
        # Check if the "correct" polynomial is still fit even if we change
        # input and output polynomial orders don't match
        coeffs, mask = lin.fit_single(dummy_path, poly_order=2, constrained=True)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 10, 10])

    def test_make_linearity_dq_nounc(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
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
        af['roman'] = dict()
        af['roman']['data'] = y_test+noise1
        af['roman']['meta'] = _meta
        af['roman']['dq'] = dq_test
        dummy_path1 = os.path.join(self.test_dir, 'dummy_img_1.asdf')
        dummy_path2 = os.path.join(self.test_dir, 'dummy_img_2.asdf')
        dummy_path3 = os.path.join(self.test_dir, 'dummy_img_3.asdf')
        af.write_to(dummy_path1)
        af['roman']['data'] = y_test+noise2
        af.write_to(dummy_path2)
        af['roman']['data'] = y_test+noise3
        af.write_to(dummy_path3)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile,
                        bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                          dtype=np.uint32))
        # Check that it works with one file
        lin.make_linearity(dummy_path1, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 10, 10], decimal=2)
        # Check that it works with 2 files  (with 2 or more files we homogeinize the data
        # to slope 1 and intercept 0 to alllow for different bias levels and count rates)
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=2, clobber=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[:, 10, 10], decimal=2)
        # Check that it works with the wrong poly order
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=5, clobber=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[3:, 10, 10], decimal=2)
        # Check 3 files
        lin.make_linearity([dummy_path1, dummy_path2, dummy_path3],
                           poly_order=2, clobber=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[:, 10, 10], decimal=2)
        # Check returning uncertainty
        lin.make_linearity(dummy_path1, poly_order=2, return_unc=True, clobber=True)
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 10, 10], decimal=2)
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=2, clobber=True,
                           return_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[:, 10, 10], decimal=2)

    def test_make_linearity_dq_useunc(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
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
        af['roman'] = dict()
        af['roman']['data'] = y_test+noise1
        af['roman']['meta'] = _meta
        af['roman']['dq'] = dq_test
        dummy_path1 = os.path.join(self.test_dir, 'dummy_img_1.asdf')
        dummy_path2 = os.path.join(self.test_dir, 'dummy_img_2.asdf')
        dummy_path3 = os.path.join(self.test_dir, 'dummy_img_3.asdf')
        af.write_to(dummy_path1)
        af['roman']['data'] = y_test+noise2
        af.write_to(dummy_path2)
        af['roman']['data'] = y_test+noise3
        af.write_to(dummy_path3)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile,
                        bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                          dtype=np.uint32))
        # Check that it works with one file
        lin.make_linearity(dummy_path1, poly_order=2, use_unc=True)
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 10, 10], decimal=2)
        # Check that it works with 2 files  (with 2 or more files we homogeinize the data
        # to slope 1 and intercept 0 to alllow for different bias levels and count rates)
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=2, clobber=True, use_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[:, 10, 10], decimal=2)
        # Check that it works with the wrong poly order
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=5, clobber=True, use_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[3:, 10, 10], decimal=2)
        # Check returning the uncertainty
        lin.make_linearity(dummy_path1, poly_order=2, clobber=True, use_unc=True, return_unc=True)
        testing.assert_almost_equal(poly_coeffs, lin.data[:, 10, 10], decimal=2)
        lin.make_linearity([dummy_path1, dummy_path2], poly_order=2, clobber=True,
                           use_unc=True, return_unc=True)
        testing.assert_almost_equal((poly_coeffs[0]/poly_coeffs[1], 1, 0),
                                    lin.data[:, 10, 10], decimal=2)

    def test_make_linearity_dq_ngrid(self):
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
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
        af['roman'] = dict()
        af['roman']['data'] = y_test+noise1
        af['roman']['meta'] = _meta
        af['roman']['dq'] = dq_test
        dummy_path1 = os.path.join(self.test_dir, 'dummy_img_1.asdf')
        dummy_path2 = os.path.join(self.test_dir, 'dummy_img_2.asdf')
        dummy_path3 = os.path.join(self.test_dir, 'dummy_img_3.asdf')
        af.write_to(dummy_path1)
        af['roman']['data'] = y_test+noise2
        af.write_to(dummy_path2)
        af['roman']['data'] = y_test+noise3
        af.write_to(dummy_path3)
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile,
                        bit_mask=np.zeros((y_test.shape[1],
                                           y_test.shape[2]),
                                          dtype=np.uint32))
        # Check that it breaks with one file
        with self.assertRaises(ValueError):
            lin.make_linearity(dummy_path1, poly_order=4, nframes_grid=3)
        with self.assertRaises(ValueError):
            lin.make_linearity([dummy_path1, dummy_path2], poly_order=4, nframes_grid=3,
                               clobber=True)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)
'''

if __name__ == '__main__':
    unittest.main()
