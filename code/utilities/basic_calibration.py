"""
Tools for doing basic calibration of reads. This includes
reference pixel subtraction and zero read subtraction.
"""

import logging
import numpy as np
from astropy import units
from ..utilities import dcl


class CalDCL:

    def __init__(self, file):

        data = dcl.read_dcl_data(file)
        self.file = file
        self.all_reads = data['reads']
        self.meta = data['meta']

        # Future arrays
        self.cal_reads = None
        self.cal_reads_units = None
        self.zero_read = None
        self.reference_pixels = None
        self.mean_reference_pixels = None
        self.ramp_image = None
        self.ramp_image_units = None

    def subtract_reference_pixels(self):

        logging.info('Calculating row-mean of reference pixels')

        # Grab the 8 columns of reference pixels from either side of the cube.
        # Compute the row average of the reference pixels.
        self.reference_pixels = np.concatenate((self.all_reads[:, :, :4], self.all_reads[:, :, -4:]), axis=2)
        new_shape = (self.all_reads.shape[0], self.all_reads.shape[1], 1)
        self.mean_reference_pixels = np.mean(self.reference_pixels, axis=2, dtype=np.int32).reshape(new_shape)

        logging.info('Row-subtracting reference pixel mean')

        # Subtract off the mean reference pixel per row.
        self.cal_reads = self.all_reads[1:] - self.mean_reference_pixels[1:]
        self.zero_read = self.all_reads[0] - self.mean_reference_pixels[0]
        self.cal_reads_units = units.adu

    def subtract_zero_read(self):

        logging.info('Subtracting zero read')

        self.cal_reads -= self.zero_read

    def ramp_fit(self):

        logging.info('Performing ramp fit')

        # Get the shape of the cube and the number of reads.
        cube_shape = self.cal_reads.shape
        data_x = [self.meta['FRTIME'] * (frame + 1) for frame in range(cube_shape[0])]

        # Reshape the data and compute slopes. Then reshape back to the
        # 2D image shape (i.e., the flattened cube).
        data_y = self.cal_reads.copy().reshape(cube_shape[0], -1)
        slopes, _ = np.polyfit(data_x, data_y, 1)
        self.ramp_image = slopes.reshape(cube_shape[1], cube_shape[2]).astype(np.float32)
        self.ramp_image_units = units.adu / units.second
