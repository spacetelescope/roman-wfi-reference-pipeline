"""
There are multiple changes in this crossover_pseudocode.py but only one is needed and is dependent on the method of implementation.

In romancal / refpix there is a _data.py script. Inside of that are several functions that assist the main reference pixel correction and the code changes in crossover_pseudocode.py will apply to that _data.py script. 

Option 1: Live Computation (Code-Based)
If we decide to implement computation of the crossover filter inside of _data.py itself we can either change the 
channel_correction definition or the classmethod from_ref. Either way the computation will be done at load time each time.

"""

def channel_correction(self, coeffs: Coefficients) -> np.ndarray:
        """
        Generator which yields the correction for each channel.
            This can then be stacked into a single array to form the correction.
        """
        # We need to multiply by 2 because we are only using half of the FFT because
        # the data is real
        normalization = coeffs.gamma.shape[1] * 2

        # --- CROSSOVER FILTER SETUP ---
        # Create a high-frequency crossover (apodization) filter to gracefully
        # push the left/right reference coefficients (gamma/zeta) to 0.0
        # above a certain threshold, preventing white noise injection.

        fft_len = coeffs.gamma.shape[1]
        pixel_freq_hz = 203125.0  # PIXEL_READ_FREQ_HZ

        # Create an array of the actual frequencies (in Hz) for each FFT bin
        freqs = np.linspace(0, pixel_freq_hz / 2.0, fft_len)

        # Define the taper bounds (matching TVAC2/Legacy notes)
        cutoff_start_hz = 1450.0
        cutoff_end_hz = 1650.0

        # Initialize filter as all 1.0s (no attenuation)
        crossover_filter = np.ones(fft_len, dtype=coeffs.gamma.dtype)

        # Apply a raised cosine taper to smoothly roll from 1.0 down to 0.0
        taper_mask = (freqs > cutoff_start_hz) & (freqs < cutoff_end_hz)
        phase = np.pi * (freqs[taper_mask] - cutoff_start_hz) / (cutoff_end_hz - cutoff_start_hz)
        crossover_filter[taper_mask] = (np.cos(phase) + 1.0) / 2.0

        # Force everything above the end frequency to exactly 0.0
        crossover_filter[freqs >= cutoff_end_hz] = 0.0
        # ------------------------------

        for gamma, zeta, alpha in coeffs:
            # Multiply gamma and zeta by the crossover filter.
            # Alpha is left untouched so high frequencies default to 1-stream correction.
            correction = (
                np.multiply(self.left, gamma * crossover_filter)
                + np.multiply(self.right, zeta * crossover_filter)
                + np.multiply(self.amp33, alpha)
            ) * normalization

            # hold onto the previous correction so that shape is maintained
            # for the blank correction for the amp33 channel
            correction = fft.irfft(correction)
            yield correction

        # Add zeros in for the amp33 channel as it does not get changed
        yield np.zeros(correction.shape)

"""
Alternative method for Option 1
"""

@classmethod
    def from_ref(cls, ref: RefpixRefModel) -> Coefficients:
        
        # --- BUILD THE CROSSOVER FILTER ONCE AT LOAD TIME ---
        fft_len = ref.gamma.shape[1]
        n_samples = (fft_len - 1) * 2
        pixel_freq_hz = 203125.0
        
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / pixel_freq_hz)

        # (If they make these reference file parameters, you'd pull them from `ref` here)
        cutoff_start_hz = 1450.0  
        cutoff_end_hz = 1650.0    

        crossover_filter = np.ones(fft_len, dtype=ref.gamma.dtype)

        taper_mask = (freqs > cutoff_start_hz) & (freqs < cutoff_end_hz)
        phase = np.pi * (freqs[taper_mask] - cutoff_start_hz) / (cutoff_end_hz - cutoff_start_hz)
        crossover_filter[taper_mask] = (np.cos(phase) + 1.0) / 2.0

        crossover_filter[freqs >= cutoff_end_hz] = 0.0
        # ----------------------------------------------------

        # Apply the filter directly to gamma and zeta before returning the class!
        return cls(
            gamma=ref.gamma * crossover_filter, 
            zeta=ref.zeta * crossover_filter, 
            alpha=ref.alpha
        )

"""
Option 2: Reference File Ingestion (CRDS-Based)
If we decide instead to load the crossover filter from the reference file then the classmethod from_ref is updated to just multiply the coefficients by the crossover filter read in from the reference file:
"""

@classmethod
    def from_ref(cls, ref: RefpixRefModel) -> Coefficients:
        """
        Ingest the coefficients from the reference file, permanently applying 
        the high-frequency crossover filter to the left and right weights.
        """
        return cls(
            gamma=ref.gamma * ref.crossover_filter, 
            zeta=ref.zeta * ref.crossover_filter, 
            alpha=ref.alpha
        )
