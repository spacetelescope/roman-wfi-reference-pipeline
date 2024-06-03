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
    data: input data array in cube shape
    wfi_type: constant string WFI_TYPE_IMAGE, WFI_TYPE_GRISM, or WFI_TYPE_PRISM
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
