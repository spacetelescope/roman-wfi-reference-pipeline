import logging
from abc import ABC

import numpy as np
from wfi_reference_pipeline.constants import (
    WFI_FRAME_TIME,
    WFI_MODE_WIM,
    WFI_MODE_WSM,
    WFI_TYPE_IMAGE,
)


class DataCube(ABC):
    """
    DataCube class to consolidate cube specific attributes.
    Classes derived from this bass class will handle common fitting methods to
    calculate cube properties, such as rate and intercept images, for reference types.

    num_i_pixels x num_j_pixels, 4096x4096 is entire detector, 1024x1024 would be 4 sub-images

    Parameters
    -------
    self.ref_type_data: input data array in cube shape
    self.wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
    """

    def __init__(self, data, wfi_type):
        self.data = data
        self.frame_time = None  # wfi_mode dependent exposure frame time per read.
        self.num_i_pixels = None  # number of pixels in 2D frames/reads, assume square pixels only, 4096x4096 is standard but not default
        self.num_j_pixels = None  # the j value of the datacube should always be the same size as the num_i_pixels.
        self.num_reads = None  # Number of reads in data.
        self.time_array = None  # Frame for each read

        # Initialize Arrays
        # Make the time array for the length of the dark read cube exposure.
        # Generate the time array depending on WFI mode.
        self.num_reads, self.num_i_pixels, self.num_j_pixels = np.shape(self.data)
        if self.num_i_pixels != self.num_j_pixels:
            raise ValueError(
                f"DataCube initialization not correct shape: ({self.num_reads}, {self.num_i_pixels}, {self.num_j_pixels})"
            )

        if wfi_type == WFI_TYPE_IMAGE:
            self.frame_time = WFI_FRAME_TIME[
                WFI_MODE_WIM
            ]  # frame time in imaging mode in seconds
        else:
            self.frame_time = WFI_FRAME_TIME[
                WFI_MODE_WSM
            ]  # frame time in spectral mode in seconds
        logging.info(
            f"Creating exposure time array {self.num_reads} reads long with a frame "
            f"time of {self.frame_time} seconds."
        )
        self.time_array = np.array(
            [self.frame_time * i for i in range(1, self.num_reads + 1)]
        )


# TODO - We can divide these into ref type specific files depending on shared routines.
# Lets keep it in one place till we have a better idea of what we are working with with the other reference types
class DarkDataCube(DataCube):
    """
    DarkDataCube class derived from DataCube.
    Handles Dark specific cube calculations
    Provide common fitting methods to calculate cube properties.

    Parameters
    -------
    self.ref_type_data: input data array in cube shape
    self.wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
    """
    def __init__(self, ref_type_data, wfi_type):
        self.ramp_model = None  # Ramp model of data cube.
        self.rate_image = None  # the slope of the fitted data_cube
        self.intercept_image = None  # the y intercept of a line fit to the data_cube

        degree = 1  # TODO how do we know what degree we should be using?
        # Inherit reference_type.
        super().__init__(
            data=ref_type_data,
            wfi_type=wfi_type,
        )


class ReadnoiseDataCube(DataCube):
    """
    ReadnoiseDataCube class derived from DataCube.
    Handles Readnoise specific cube calculations
    Provide common fitting methods to calculate cube properties, such as rate and intercept images, for reference types.

    Parameters
    -------
    self.ref_type_data: input data array in cube shape
    self.wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
    """

    def __init__(self, ref_type_data, wfi_type):
        self.ramp_model = None  # Ramp model of data cube.
        self.rate_image = None  # the slope of the fitted data_cube
        self.intercept_image = None  # the y intercept of a line fit to the data_cube

        degree = 1  # TODO how do we know what degree we should be using?
        # Inherit reference_type.
        super().__init__(
            data=ref_type_data,
            wfi_type=wfi_type,
        )
        try:
            coeffs_array, covars_array = np.polyfit(
                self.time_array,
                ref_type_data.reshape(len(self.time_array), -1),
                degree,
                full=False,
                cov=True,
            )
            # Reshape the parameter slope array into a 2D rate image.
            self.rate_image = coeffs_array[0].reshape(
                self.num_i_pixels, self.num_j_pixels
            )
            # Reshape the parameter y-intercept array into a 2D image.
            self.intercept_image = coeffs_array[1].reshape(
                self.num_i_pixels, self.num_j_pixels
            )
        except (TypeError, ValueError) as e:
            logging.error(f"Unable to initialize ReadnoiseDataCube with error {e}")
            # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error

        _fit_data_cube_ramp_model(self, coeffs_array, order=degree)


def _fit_data_cube_ramp_model(data_cube, coeffs_array, order=1):
    """
    fit_ramp_model performs a linear or quadratic fit to the input read cube for each pixel. The slope
    and intercept are calculated along with the covariance matrix which has the corresponding diagonal error
    estimates for variances in the model fitted parameters.

    Save to attribute rate_image, intercept_image, and ramp_model.

    Currently used for ReadnoiseDataCube

    NOTE: Keep covariance matrices in code for future use determination.
    TODO - Algorithm on how to incorporate "order"?

    """
    logging.info("Making ramp model for the input read cube.")
    # Reshape the 2D array into a 1D array for input into np.polyfit().
    # The model fit parameters p and covariance matrix v are returned.
    try:
        # Reshape the returned covariance matrix slope fit error.
        # rate_var = v[0, 0, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO -VERIFY USE
        # returned covariance matrix intercept error.
        # intercept_var = v[1, 1, :].reshape(data_cube.num_i_pixels, data_cube.num_j_pixels) TODO - VERIFY USE

        data_cube.ramp_model = np.zeros(
            (data_cube.num_reads, data_cube.num_i_pixels, data_cube.num_j_pixels),
            dtype=np.float32,
        )

        if order == 1:
            # y = m * x + b
            # where y is the pixel value for every read,
            # m is the slope at that pixel or the rate image,
            # x is time (this is the same value for every pixel in a read)
            # b is the intercept value or intercept image.
            for tt in range(0, len(data_cube.time_array)):
                data_cube.ramp_model[tt, :, :] = (
                    data_cube.rate_image * data_cube.time_array[tt]
                    + data_cube.intercept_image
                )
        elif order == 2:
            # y = ax^2 + bx + c
            # where we dont have a single rate image anymore, we have coefficients
            for tt in range(0, len(data_cube.time_array)):
                a, b, c = coeffs_array
                data_cube.ramp_model[tt, :, :] = (
                    a * data_cube.time_array[tt] ** 2 + b * data_cube.time_array[tt] + c
                )

        else:
            raise ValueError("This function only supports polynomials of order 1 or 2.")

    except (ValueError, TypeError) as e:
        logging.error(f"Unable to make_ramp_cube_model with error {e}")
        # TODO - DISCUSS HOW TO HANDLE ERRORS LIKE THIS, ASSUME WE CAN'T JUST LOG IT - For cube class discussion - should probably raise the error
