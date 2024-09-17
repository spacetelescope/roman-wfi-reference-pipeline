'''
Various functions that are common to multiple IRRC steps performed in irrc_extract_ramp_sums and irrc_generate_weights

Written by (Rauscher et al., in prep):
    - S. Maher
    - R. Arendt
'''

import threading

import numpy as np
import scipy.fft as spfft

from .irrc_constants import NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD, NUM_OUTPUT_CHANS, \
    ALL_CHAN_RANGE, REFERENCE_ROWS, \
    NUM_COLS, NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN

import logging
logger = logging.getLogger('ReferencePixel util')

    
def remove_linear_trends(data_frames_rowscols:np.ndarray, subtract_offset_only:bool):
    '''
    Remove linear trends (gains/slope, biases/offsets) per pixel 

    Parameters
    ----------
    data_frames_rowscols: ndarray
        image data [num_frames][num_rows][num_cols].  The data is modified IN PLACE
    subtract_offset_only: boolean
        if True, only the liner model offset is removed.  If False, the offset and slope are removed
    
    Returns
    ----------
    per-pixel image array of modeled m (slope) and b (offset) 
    '''

    num_frames, num_rows, num_cols = data_frames_rowscols.shape

    # for mat multiply
    data_frames_rowscols = data_frames_rowscols.reshape((num_frames, num_cols * num_rows))

    frame_ramp = np.arange(num_frames, dtype=data_frames_rowscols.dtype)

    # subtract mean(frame_ramp_minus_mean) to minimize correlation of slope and offset
    frame_ramp_minus_mean = frame_ramp - np.mean(frame_ramp)

    sxy = np.matmul(frame_ramp_minus_mean, data_frames_rowscols, dtype=data_frames_rowscols.dtype)
    sx = np.sum(frame_ramp_minus_mean)
    sxx = np.sum(frame_ramp_minus_mean ** 2)
    sy = np.sum(data_frames_rowscols, axis=0)

    m = (num_frames * sxy - sx * sy) / (num_frames * sxx - sx ** 2)
    b = (sy * sxx - sxy * sx) / (num_frames * sxx - sx ** 2)

    for frame in range(num_frames):
        if not subtract_offset_only:
            data_frames_rowscols[frame,:] -= frame_ramp_minus_mean[frame] * m + b  # subtract slope and offset
        else:
            data_frames_rowscols[frame,:] -= b  # subtract offset

    data_frames_rowscols = data_frames_rowscols.reshape(num_frames, num_rows, num_cols)
    return m.reshape(num_rows, num_cols), b.reshape(num_rows, num_cols)


def remove_linear_trends_per_frame(logger:logging.Logger, data_chans_frames_rowschancols:np.ndarray, subtract_offset_only:bool, multithread=True):
    '''
    Entry point for Fitting and removal of slopes per frame to remove issues at frame boundaries. 

    Parameters
    ----------
    logger: logging.Logger
        logger
    data_chans_frames_rowschancols: ndarray
        [numOutChannels(33)][num_frames][num_rows][numColsWithPadding].  The data is modified IN PLACE
    subtract_offset_only: boolean
        if True, only the liner model offset is removed.  If False, the offset and slope are removed
    multithread: boolean, default = True
        should multithreading be used in various calculations?
    '''
    _, num_frames, num_rows, _ = data_chans_frames_rowschancols.shape
    time_domain_range = np.arange(NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * num_rows, dtype=data_chans_frames_rowschancols.dtype).reshape(num_rows, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD)
    time_domain_range = time_domain_range - time_domain_range.mean()

    exec_channel_func_threads(ALL_CHAN_RANGE, remove_linear_trends_per_frame_chan_func,
            (logger, num_frames, time_domain_range, data_chans_frames_rowschancols, subtract_offset_only), multithread=multithread)


def remove_linear_trends_per_frame_chan_func(chan_number:int, logger:logging.Logger, num_frames:int,
    time_domain_range:np.ndarray, data_chans_frames_rowschancols:np.ndarray, subtract_offset_only:bool):
    '''
    Per-channel function for fitting and removal of slopes per frame to remove issues at frame boundaries - single channel only

    Parameters
    ----------
    chan_number: int
    logger: logging.Logger
    num_frames: int
    time_domain_range: ndarray
    data_chans_frames_rowschancols: ndarray
        [numOutChannels(33)][num_frames][num_rows][numColsWithPadding].  The data is modified IN PLACE
    subtract_offset_only: boolean
        if True, only the liner model offset is removed.  If False, the offset and slope are removed
    '''

    ab_3 = np.zeros((NUM_OUTPUT_CHANS, num_frames, 2), dtype=np.float64)  # for top+bottom ref pixel rows
    td_ref = time_domain_range[REFERENCE_ROWS,:]

    for frame in range(num_frames):
        x_vals = td_ref[(data_chans_frames_rowschancols[chan_number, frame, REFERENCE_ROWS,:] != 0)]  # where data[chan_number,frame,row4plus4,:] != 0)
        data2 = data_chans_frames_rowschancols[chan_number, frame, REFERENCE_ROWS,:]
        y_vals = data2[(data2 != 0)]
        if x_vals.size < 1 or y_vals.size < 1:
            logger.warn(f'Skipping empty data section.  Frame={frame}, Chan={chan_number}')
            continue

        ab_3[chan_number, frame,:] = np.polyfit(x_vals, y_vals, 1)

        data_notzero = (data_chans_frames_rowschancols[chan_number, frame,:,:] != 0)

        if not subtract_offset_only:
            data2 = (time_domain_range * ab_3[chan_number, frame, 0] + ab_3[chan_number, frame, 1])
        else:
            data2 = ab_3[chan_number, frame, 1]

        data_chans_frames_rowschancols[chan_number, frame,:,:] -= data2 * data_notzero


def interp_zeros_channel_fun(chan_number:int, interp_func:np.ndarray, data_chans_frames_rowscols:np.ndarray, num_frames:int, num_rows:int, num_cols_per_output_chan_with_pad:int):
    '''
    Convolve interp_func across pixels with values of 0 for a single channel.  We are not using the outlier mask and thus possibly interpolating
    pixels with original values of 0.  This is to ensure we don't have discontinuities (around 0) when doing the FFT interpolation, which could
    damage the performance.
    
    Parameters
    ----------
    chan_number: int
    interp_func: ndarray
    data_chans_frames_rowscols: ndarray
        Data to interpolate, which is done IN PLACE
    num_frames: int
    num_rows: int
    num_cols_per_output_chan_with_pad: int
    '''
    for frame in range(num_frames):
        dat = np.reshape(data_chans_frames_rowscols[chan_number, frame,:,:], num_cols_per_output_chan_with_pad * num_rows)

        # Apply to areas with 0 value.
        apply_mask = (dat != 0).astype(int)

        data_convolve = np.convolve(dat, interp_func, mode='same')  # 'same' -> /edge_wrap in IDL
        apply_mask_convolve = np.convolve(apply_mask, interp_func, mode='same')

        sfm3 = data_convolve / apply_mask_convolve
        dat = np.reshape(sfm3, (num_rows, num_cols_per_output_chan_with_pad))
        mask2d = np.reshape(apply_mask, (num_rows, num_cols_per_output_chan_with_pad))
        data_chans_frames_rowscols[chan_number, frame,:,:] += dat * (1 - mask2d)
        # Repair any NaNs
        np.nan_to_num(data_chans_frames_rowscols[chan_number,:,:,:], copy=False)


def fft_interp_step_channel_fun(chan_number:int, data_chans_frames_flattenedimage:np.ndarray, data_fft_out:np.ndarray, read_only_pixels_are_onemask_chanrowcol:np.ndarray, appodize_func:np.ndarray, num_fft_iterations:int):
    '''
    Perform FFT interpolation on pixels for a single channel. A final FFT is done before return (needed by IRRC algorithm and more performant to do here) 

    Parameters
    ----------
    chan_number: int
    data_chans_frames_flattenedimage: ndarray
        (channels, frames, rows*cols) of data.  WILL BE UPDATED IN PLACE
    data_fft_out: ndarray
        Returned interpolated data - with a final FFT applied 
    read_only_pixels_are_onemask_chanrowcol: ndarray
        image mask with 1's represent read-only pixels (e.g., 0 entries will be updated with interpolation)
    appodize_func: ndarray
    num_fft_iterations: int
    '''
    chandata_framesflat = data_chans_frames_flattenedimage[chan_number,:,:]
    read_only_pixels_are_ones_framemask_rowscols_padded = np.zeros((NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD), dtype=bool)
    read_only_pixels_are_ones_framemask_rowscols_padded[:,:NUM_COLS_PER_OUTPUT_CHAN] = read_only_pixels_are_onemask_chanrowcol[chan_number,:,:]

    fft_interp(chandata_framesflat, read_only_pixels_are_ones_framemask_rowscols_padded.flatten(), appodize_func, num_fft_iterations)

    data_fft_out[chan_number,:,:] = spfft.rfft(chandata_framesflat / chandata_framesflat[0].size)


def fft_interp(chandata_framesflat:np.ndarray, read_only_pixels_are_onemask_flattenedimage:np.ndarray, apodize_filter:np.ndarray, num_iterations:int):
    '''
    Perform FFT interpolation on pixels not marked as read-only

    Parameters
    ----------
    chandata_framesflat: ndarray
        Multi-frame, single channel data with each frame flattened to 1D.  WILL BE UPDATED IN PLACE
    read_only_pixels_are_onemask_flattenedimage: ndarray  
        Flattened image size mask indicating (good) pixels that should be used in the interpolation but not modified
    apodize_filter: ndarray
    num_iterations: int
        number of times to iterate the interpolation
    '''

    num_frames = len(chandata_framesflat)
    read_only_pixels_indices_flat = np.where(read_only_pixels_are_onemask_flattenedimage)[0]

    for frame in range(num_frames):

        chan_framedata_flat = chandata_framesflat[frame]
        read_only_pixels_values_flat = chan_framedata_flat[read_only_pixels_indices_flat]

        for _ in range(num_iterations):

            fft_result = apodize_filter * spfft.rfft(chan_framedata_flat, workers=1) / chan_framedata_flat.size
            chan_framedata_flat = spfft.irfft(fft_result * chan_framedata_flat.size, workers=1)

            # Return read only pixels
            chan_framedata_flat[read_only_pixels_indices_flat] = read_only_pixels_values_flat

        chandata_framesflat[frame,:] = chan_framedata_flat


def get_reference_mask(value_for_refpixels:bool=0):
    '''
    Generate a full image mask of the reference pixels
    
    Parameters
    ----------
    value_for_refpixels: boolean, default = 0
        set reference pixels to this value
    '''

    if value_for_refpixels:
        mask = np.zeros((NUM_ROWS, NUM_COLS), dtype=bool)
    else:
        mask = np.ones((NUM_ROWS, NUM_COLS), dtype=bool)

    # set reference pixels to 0
    mask[:, 0:4] = value_for_refpixels
    mask[:, 4092:4096] = value_for_refpixels
    mask[0:4, 0:4096] = value_for_refpixels
    mask[-4:, 0:4096] = value_for_refpixels
    return mask

def get_fft_apodize_function():
    
    apo_len = NUM_ROWS * NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD
    apo2 = np.abs(np.fft.rfftfreq(apo_len, 1 / apo_len))
    apodize_func = np.cos(2 * np.pi * apo2 / apo_len) / 2.0 + 0.5 
    return apodize_func

def get_trig_interpolation_function(data):
    return np.sin(np.arange(NUM_OUTPUT_CHANS, dtype=data.dtype) / 32 * np.pi)[1:]

def exec_channel_func_threads(chan_index_range:range, target_func, func_args, multithread=False):
    '''
    Execute a function over a range of channels.  If multithread is True, all executions
    will be executed on individual threads.  
    
    Parameters
    ----------
    chan_index_range: range
        E.g., 'range(NUM_OUTPUT_CHANS')
    target_func: 
        function to call, first arg must be channel number
    func_args: 
        function argument tuple NOT INCLUDING channel number e.g. (sig_data, goodPixelsOneIsMask_RowCol)
    multithread: boolean, default = False
        if True allocate a separate thread for each channel; otherwise the channels are processed sequentially in a single thread
    '''
    
    if multithread:
        logger.info("multithreading-> ")
        thread_list = []
        for c in chan_index_range:
            func_args_with_chan = (c,) + func_args
            thread_list.append(threading.Thread(target=target_func, args=func_args_with_chan))

        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()
    else:
        logger.info("Single threading-> ")
        for c in chan_index_range:
            func_args_with_chan = (c,) + func_args
            target_func(*func_args_with_chan)
    logger.info('Done executing over range of channels')