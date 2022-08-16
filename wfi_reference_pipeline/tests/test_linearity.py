import unittest
import numpy as np
import numpy.testing as testing
# from numpy import random
import tempfile
import asdf
from astropy.time import Time
from ..linearity.linearity import Linearity
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


class LinearityTestCase(unittest.TestCase):
    def test_make_linearity_dummy(self):
        """
        Test the method make_linearity() if no input is
        passed,, and a dummy reference file is created
        """
        # Prepare the metadata that needs to be passed to Linearity
        _meta = setup_dummy_meta()

        dummy_img = np.zeros((1, 1, 1))
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(dummy_img, _meta, outfile=outfile)
        with testing.assert_warns(Warning):
            lin.make_linearity(dummy_img)

        # Check that the file is created
        assert(os.path.exists(lin.outfile))

        # Check that the file is not overwritten
        with self.assertRaises(FileExistsError):
            lin.make_linearity(dummy_img)

    def test_make_linearity_single(self):
        """
        Test fit to single linearity file
        """

        # Set up a dummy file
        _meta = setup_dummy_meta()
        af = asdf.AsdfFile()
        # Set up a grid of times
        t_test = np.linspace(0, 10, 20)
        # Set up a polynomial
        poly_coeffs = (0.1, 10, 0.001)
        y_test = np.polyval(poly_coeffs, t_test)
        y_test = y_test[:, None, None]*np.ones((40, 40))  # Test with a (20, 40, 40) array
        _meta['exposure'] = dict()
        _meta['exposure']['frame_time'] = t_test
        af['roman'] = dict()
        af['roman']['data'] = y_test
        af['roman']['meta'] = _meta
        dummy_path = os.path.join(self.test_dir, 'dummy_img.asdf')
        af.write_to(dummy_path)
        # Run without noise
        outfile = os.path.join(self.test_dir, 'roman_linearity.asdf')
        lin = Linearity(y_test, _meta, outfile=outfile)
        coeffs, mask = lin.fit_single(dummy_path, poly_order=2)
        testing.assert_almost_equal(poly_coeffs, coeffs[:, 0, 0])

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(dir='./',
                                         prefix='TestWFIrefpipe-')

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
