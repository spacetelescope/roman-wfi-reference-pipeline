"""
Tools for doing basic calibration of reads. This includes
reference pixel subtraction and zero read subtraction.
"""

import numpy as np
from ..utilities import dcl


class CalDCL:

    def __init__(self, file):

        data = dcl.read_dcl_data(file)
        self.file = file
        self.all_reads = data['reads']
        self.meta = data['meta']

        # Future arrays
        self.cal_reads = None
        self.zero_read = None
        self.reference_pixels = None
        self.mean_reference_pixels = None
        self.cube_shape = self.all_reads.shape

    def subtract_reference_pixels(self):

        # Grab the 8 columns of reference pixels from either side of the cube.
        self.reference_pixels = np.concatenate((self.all_reads[:, :, :4], self.all_reads[:, :, -4:]), axis=2)
        mean_reference_pixels = np.mean(self.reference_pixels, axis=2, dtype=np.int32)
        self.mean_reference_pixels = mean_reference_pixels.reshape(self.cube_shape[0], self.cube_shape[1], 1)

        all_cal_reads = self.all_reads - self.mean_reference_pixels
        self.cal_reads = all_cal_reads[1:]
        self.zero_read = all_cal_reads[0]

    def subtract_zero_read(self):

        self.cal_reads -= self.zero_read
