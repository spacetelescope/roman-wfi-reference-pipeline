import numpy as np
from astropy.stats import sigma_clipped_stats


class ReadNoise:

    def __init__(self, zero1, zero2):

        self.zero1 = zero1
        self.zero2 = zero2

        # Future arrays
        self.read_noise = None

    def get_read_noise(self):

        delta = self.zero1 - self.zero2
        _, noise, _ = sigma_clipped_stats(delta, sigma=5, maxiters=2, axis=1)
        variance = noise**2

        # Linear fit to the data. Could get fancy with this later to do
        # outlier rejection...
        var_func = np.poly1d(np.polyfit(range(delta.size[1]),
                                        variance[::-1] / 2, 1))
        self.read_noise = np.sqrt(var_func(0))
