'''
IRRC Constants : various constants that do not change for WFI detectors

Written by (Rauscher et al., in prep):
    - S. Maher
    - R. Arendt

'''

NUM_ROWS=4096

# CHANS = amplifiers = outputs
NUM_OUTPUT_CHANS = 33
NUM_COLS_PER_OUTPUT_CHAN = 128
NUM_COLS = NUM_OUTPUT_CHANS * NUM_COLS_PER_OUTPUT_CHAN

# END_OF_ROW_PIXEL_PAD is the effective number of pixels sampled during the pause at the end of 
# each NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD. The padding is needed to preserve phase of temporally periodic signals.
END_OF_ROW_PIXEL_PAD = 12

NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD = NUM_COLS_PER_OUTPUT_CHAN + END_OF_ROW_PIXEL_PAD

ALL_CHAN_RANGE = range(NUM_OUTPUT_CHANS)

# The 33rd amplifier is a reference amplifier.  python is 0-indexed, so this is index 32. 
REFERENCE_CHAN = 32

# Range of non-reference (normal) channels.  Warning this definition only works because REFERENCE_CHAN is 
# the last channel!
CHAN_RANGE_WITHOUT_REFERENCES = range(NUM_OUTPUT_CHANS - 1)

# reference columns are included in amplifiers 1 and 32 (or 0 and 31 for python 0-indexing)
REFERENCE_ROWS = [0, 1, 2, 3, 4092, 4093, 4094, 4095]

COEFF_SHAPE = (NUM_OUTPUT_CHANS - 1, int(NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS / 2) + 1)

PIXEL_READ_FREQ_HZ = 203.125e3

#  Calculate normalization factor for reference pixel reference streams.
#  At the end of the code, sums invloving the refence pixels are multiplied 
#  by this factor to account for the the fact that only 4 pixels per row are contributing signal.
#  This makes the reference pixel power spectra comparable to the data and reference output
#  at low frequency, and will result in gamma and zeta weights closer to untity 
#  in part B (weight calculation).
REFPIX_NORM = NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD/4.0