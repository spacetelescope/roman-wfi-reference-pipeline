import numpy as np


def make_mask():

    mask = np.zeros((4096, 4096), dtype=np.uint32)

    # Add in reference pixels (DQ bit 31).
    mask[:4, :] = 2**31
    mask[-4:, :] = 2**31
    mask[:, 4:] = 2**31
    mask[:, -4:] = 2**31

    return mask
