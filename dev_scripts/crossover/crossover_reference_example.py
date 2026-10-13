import numpy as np

# 1. Pipeline parameters (Roman WFI)
fft_len = 286721
n_samples = (fft_len - 1) * 2  # 573,440 pixels
pixel_freq_hz = 203125.0
cutoff_start_hz = 1450.0
cutoff_end_hz = 1650.0

# 2. Build the array
freqs = np.fft.rfftfreq(n_samples, d=1.0 / pixel_freq_hz)
crossover_filter = np.ones(fft_len, dtype=float)

taper_mask = (freqs > cutoff_start_hz) & (freqs < cutoff_end_hz)
phase = np.pi * (freqs[taper_mask] - cutoff_start_hz) / (cutoff_end_hz - cutoff_start_hz)
crossover_filter[taper_mask] = (np.cos(phase) + 1.0) / 2.0
crossover_filter[freqs >= cutoff_end_hz] = 0.0

# 3. Save the full array to a text file
output_filename = "crossover_filter_reference.txt"

with open(output_filename, "w") as f:
    # Write the header metadata
    f.write("# Proposed 1D Array for ASDF Reference File (`crossover_filter`)\n")
    f.write(f"# Shape: ({fft_len},)\n")
    f.write(f"# {'Array Index':<15} | {'Frequency (Hz)':<18} | {'Filter Weight'}\n")
    f.write("-" * 55 + "\n")
    
    # Write all 286,721 rows
    for idx in range(fft_len):
        f.write(f"{idx:<15} | {freqs[idx]:<18.2f} | {crossover_filter[idx]:<25.6f}\n")

print(f"Success: Full filter array saved to '{output_filename}'!")