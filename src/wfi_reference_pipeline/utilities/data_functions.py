import numpy as np

from wfi_reference_pipeline.constants import DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT, VIRTUAL_PIXEL_DEPTH


def get_science_pixels_cube(data_cube, border=VIRTUAL_PIXEL_DEPTH):
    num_reads, num_i_pixels, num_j_pixels = np.shape(data_cube)
    if num_i_pixels == num_j_pixels == 4096:
        return data_cube[:,border:-border,border:-border]
    elif num_i_pixels == num_j_pixels == 4088:
        return data_cube
    else:
        raise ValueError(
            f"DataCube not correct shape: ({num_reads}, {num_i_pixels}, {num_j_pixels})"
        )

def add_science_pixels_to_resource_slice (data_cube_slice, border=VIRTUAL_PIXEL_DEPTH):
        """Places the inner data back into a new array with original shape, keeping the border."""
        resource_slice = np.zeros((4096, 4096), dtype=np.float32)
        resource_slice[border:-border, border:-border] = data_cube_slice
        return resource_slice