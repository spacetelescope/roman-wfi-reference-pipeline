import logging

import numpy as np
from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_MODE_WSM,
    WFI_TYPE_IMAGE,
)


class DataCubeModel:
    """
    DataCubeModel class to reduce cubes into images for reference types

    # fit_cube(self, order=1)
    # fit cube model (function), slope_image[][], intercept_image[][], slope_error_image[], intercept_error_image[][].
    # make_cube_model() no params or input yet. -> .cube_model


    Returns
    -------
    self.meta_data: object;
        Reference type specific meta data object.
    self.file_list: attribute;

    self.data_array: attribute;
        Class dependent variable assigned as attribute. Intended to be list of files or numpy array.
    self.ancillary: attribute;
        Other data for WFI such as filter names, frame times, WFI mode.
    self.dqflag_defs:
    """

    def __init__(self, data_cube, wfi_type):
        self.data_cube = data_cube
        self.frame_time = None  # wfi_mode dependent exposure frame time per read.
        self.nr_pixels = None  # .ni - number of pixels in 2D frames/reads, assume square pixels only, 4096x4096 is standard but not default
        self.nr_reads = None  # Number of reads in data.
        self.ramp_model = None  # Ramp model of data cube.
        self.time_array = None  # WIM/WSM dependent time array populated by the number of reads and multiplied by mode specific frame time for each read - TODO not currently used

        self.rate_image = None  # the slope of the fitted data_cube
        self.intercept_image = None  # the y intercept of a line fit to the data_cube

        # Initialize Arrays
        # Make the time array for the length of the dark read cube exposure.
        # Generate the time array depending on WFI mode.
        self.nr_reads, self.nr_pixels, _ = np.shape(self.data_cube)
        if wfi_type == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[
                WFI_MODE_WIM
            ]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[
                WFI_MODE_WSM
            ]  # frame time in spectral mode in seconds
        logging.info(
            f"Creating exposure time array {self.nr_reads} reads long with a frame "
            f"time of {self.frame_time} seconds."
        )
        self.time_arr = np.array(
            [self.frame_time * i for i in range(1, self.nr_reads + 1)]
        )

        self._fit_ramp_model()

    def _fit_ramp_model(self, order=1):
        """
        _fit_ramp_model performs a linear fit to the input read cube for each pixel. The slope
        and intercept are calculated along with the covariance matrix which has the corresponding diagonal error
        estimates for variances in the model fitted parameters.

        Save to attribute rate_image, intercept_image, and ramp_model.

        NOTE: Keep covariance matrices in code for future use determination.
        TODO - Algorithm on how to incorporate "order"?

        """
        logging.info("Making ramp model for the input read cube.")
        # Reshape the 2D array into a 1D array for input into np.polyfit().
        # The model fit parameters p and covariance matrix v are returned.
        try:
            p, v = np.polyfit(
                self.time_arr,
                self.data_cube.reshape(len(self.time_arr), -1),
                1,
                full=False,
                cov=True,
            )
            # Reshape the parameter slope array into a 2D rate image.
            self.rate_image = p[0].reshape(self.nr_pixels, self.nr_pixels)
            # Reshape the parameter y-intercept array into a 2D image.
            self.intercept_image = p[1].reshape(self.nr_pixels, self.nr_pixels)
            # Reshape the returned covariance matrix slope fit error.
            # rate_var = v[0, 0, :].reshape(self.nr_pixels, self.nr_pixels) TODO -VERIFY USE
            # returned covariance matrix intercept error.
            # intercept_var = v[1, 1, :].reshape(self.nr_pixels, self.nr_pixels) TODO - VERIFY USE

            self.ramp_model = np.zeros(
                (self.n_reads, self.nr_pixels, self.nr_pixels), dtype=np.float32
            )
            for tt in range(0, len(self.time_arr)):
                # Construct a simple linear model y = m*x + b.
                self.ramp_model[tt, :, :] = (
                    self.rate_image * self.time_arr[tt] + self.intercept_image
                )

        except (ValueError, TypeError) as e:
            logging.error(f"Unable to make_ramp_cube_model with error {e}")
            # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error
