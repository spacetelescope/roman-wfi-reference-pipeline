'''
Various functions that are common to multiple IRRC steps

@author: smaher
@author: rarendt
'''

import threading

import numpy as np
import scipy.fft as spfft
import asdf
import h5py
from astropy.io import fits

from .irrc_constants import NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD, NUM_OUTPUT_CHANS, \
    ALL_CHAN_RANGE, REFERENCE_ROWS, \
    NUM_COLS, NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN

import logging
logger = logging.getLogger('ReferencePixel util')

def read_roman_file(file_name:str, skip_first_frame:bool):
    if '.h5' in file_name:
        return read_roman_hdf5(file_name, skip_first_frame)
    elif '.fits' in file_name:
        return read_roman_fits(file_name, skip_first_frame)
    elif '.asdf' in file_name:
        return read_roman_asdf(file_name, skip_first_frame)
    else:
        raise ValueError('can only read in .h5, .asdf, or .fits.')

def read_roman_fits(file_name:str, skip_first_frame:bool):
    '''
    Read a ROMAN FITS file
    :param file_name: FITS file name
    :param skip_first_frame: optionally skip first frame
    :param logger: e.g., irrc.logger.get_logger(name)
    :returns FITS image array
    '''

    logger.info(f"Trying to read FITS file: {file_name}")
    # Cannot use a memory-mapped image: BZERO/BSCALE/BLANK header keywords present.
    with fits.open(file_name, memmap=False) as hdu_list:

        hdu_data = hdu_list[1]
        logger.info(f"Reading FITS data of dimensions {hdu_data.header['NAXIS1']}  x  {hdu_data.header['NAXIS2']} x {hdu_data.header['NAXIS3']}")

        # numpy.ndarray of type uint16 then float64 after multiplication by gain
        # FITS has an extra dimension, hence the data[0]
        # Also remove the first frame due to reset anomalies
        if skip_first_frame:
            data0 = hdu_data.data[0, 1:]
        else:
            data0 = hdu_data.data[0,:]

    data_shape = data0.shape
    num_cols = data_shape[2]
    num_rows = data_shape[1]

    if num_cols != NUM_COLS or num_rows != NUM_ROWS:
        raise Exception("File:", file_name, " has incorrect dimensions.  Expecting num_rows =", NUM_ROWS, ", num_cols =", NUM_COLS)

    # Convert from uint16 to prepare for in-place computations
    return data0.astype(np.float64)

def read_roman_hdf5(file_name:str, skip_first_frame:bool):
    '''
    read a ROMAN hdf5 file
    written by Sarah Betti based on read_roman_fits()
    returns a image array
    '''
    logger.info(f"Trying to read HDF5 file: {file_name}")

    fil = h5py.File(file_name, 'r')
    # get data
    dset = fil['Frames']
    if skip_first_frame:
        data0 = np.array(list(dset))[1:]
    else:
        data0 = np.array(list(dset))

    data_shape = data0.shape
    num_cols = data_shape[2]
    num_rows = data_shape[1]

    if num_cols != NUM_COLS or num_rows != NUM_ROWS:
        raise Exception("File:", file_name, " has incorrect dimensions.  Expecting num_rows =", NUM_ROWS, ", num_cols =", NUM_COLS)
    
    # Convert from uint16 to prepare for in-place computations
    return data0.astype(np.float64)


def read_roman_asdf(file_name:str, skip_first_frame:bool):
    '''
    read a ROMAN ASDF file
    written by Sarah Betti based on read_roman_fits()
    returns a image array
    '''
    logger.info(f"Trying to read ASDF file: {file_name}")

    fil = asdf.open(file_name)
    # get data
    data = np.array(fil.tree['roman']['data'].value)
    # get amp33
    amp33 = np.array(fil.tree['roman']['amp33'].value)
    # combine back together > amp33 added to end of array. 
    dset = np.concatenate([data, amp33], axis=2)
    if skip_first_frame:
        data0 = dset[1:]

    data_shape = data0.shape
    num_cols = data_shape[2]
    num_rows = data_shape[1]

    if num_cols != NUM_COLS or num_rows != NUM_ROWS:
        raise Exception("File:", file_name, " has incorrect dimensions.  Expecting num_rows =", NUM_ROWS, ", num_cols =", NUM_COLS)
    
    # Convert from uint16 to prepare for in-place computations
    return data0.astype(np.float64)

    
def remove_linear_trends(data_frames_rowscols:np.ndarray, subtract_offset_only:bool):
    '''
    Remove linear trends (gains/slope, biases/offsets) per pixel 
    :param data_frames_rowscols: image data [num_frames][num_rows][num_cols].  The data is modified IN PLACE
    :param subtract_offset_only: if True, only the liner model offset is removed.  If False, the offset and slope are removed
    :return per-pixel image array of modeled m (slope) and b (offset) 
    '''

    num_frames = len(data_frames_rowscols)
    num_rows = len(data_frames_rowscols[0])
    num_cols = len(data_frames_rowscols[0][0])

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
    :param data_chans_frames_rowschancols: [numOutChannels(33)][num_frames][num_rows][numColsWithPadding].  The data is modified IN PLACE
    :param subtract_offset_only: if True, only the liner model offset is removed.  If False, the offset and slope are removed
    '''

    num_frames = len(data_chans_frames_rowschancols[0])
    num_rows = len(data_chans_frames_rowschancols[0][0])
    time_domain_range = np.arange(NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * num_rows, dtype=data_chans_frames_rowschancols.dtype).reshape(num_rows, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD)
    time_domain_range = time_domain_range - time_domain_range.mean()

    exec_channel_func_threads(ALL_CHAN_RANGE, remove_linear_trends_per_frame_chan_func,
            (logger, num_frames, time_domain_range, data_chans_frames_rowschancols, subtract_offset_only), multithread=multithread)


def remove_linear_trends_per_frame_chan_func(chan_number:int, logger:logging.Logger, num_frames:int,
    time_domain_range:np.ndarray, data_chans_frames_rowschancols:np.ndarray, subtract_offset_only:bool):
    '''
    Per-channel function for fitting and removal of slopes per frame to remove issues at frame boundaries - single channel only
    :param chan_number:
    :param num_frames:
    :param time_domain_range:
    :param data_chans_frames_rowschancols: [numOutChannels(33)][num_frames][num_rows][numColsWithPadding].  The data is modified IN PLACE
    :param subtract_offset_only: if True, only the liner model offset is removed.  If False, the offset and slope are removed
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
    
    :param chan_number:
    :param interp_func:
    :param data_chans_frames_rowscols:  Data to interpolate, which is done IN PLACE
    :param num_frames:
    :param num_rows:
    :param num_cols_per_output_chan_with_pad:
    '''
    for frame in range(num_frames):
        dat = np.reshape(data_chans_frames_rowscols[chan_number, frame,:,:], num_cols_per_output_chan_with_pad * num_rows)

        # Apply to areas with 0 value.
        apply_mask = np.where(dat != 0, 1, 0)

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
    :param chan_number:
    :param data_chans_frames_flattenedimage: (channels, frames, rows*cols) of data.  WILL BE UPDATED IN PLACE
    :param data_fft_out: Returned interpolated data - with a final FFT applied 
    :param read_only_pixels_are_onemask_chanrowcol: image mask with 1's represent read-only pixels (e.g., 0 entries will be updated with interpolation)
    :param appodize_func:
    :param num_fft_iterations:
    '''
    chandata_framesflat = data_chans_frames_flattenedimage[chan_number,:,:]
    read_only_pixels_are_ones_framemask_rowscols_padded = np.zeros((NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD), dtype=bool)
    read_only_pixels_are_ones_framemask_rowscols_padded[:,:NUM_COLS_PER_OUTPUT_CHAN] = read_only_pixels_are_onemask_chanrowcol[chan_number,:,:]

    fft_interp(chandata_framesflat, read_only_pixels_are_ones_framemask_rowscols_padded.flatten(), appodize_func, num_fft_iterations)

    data_fft_out[chan_number,:,:] = spfft.rfft(chandata_framesflat / chandata_framesflat[0].size)


def fft_interp(chandata_framesflat:np.ndarray, read_only_pixels_are_onemask_flattenedimage:np.ndarray, apodize_filter:np.ndarray, num_iterations:int):
    '''
    Perform FFT interpolation on pixels not marked as read-only
    :param chandata_framesflat: Multi-frame, single channel data with each frame flattened to 1D.  WILL BE UPDATED IN PLACE
    :param read_only_pixels_are_onemask_flattenedimage:  Flattened image size mask indicating (good) pixels that should be used in the interpolation but not modified
    :param num_frames: number of frames in chandata_framesflat
    :param apodize_filter:
    :param num_iterations: number of times to iterate the interpolation
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
    
    :param value_for_refpixels: set reference pixels to this value
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
    
    :param chan_index_range: E.g., 'range(NUM_OUTPUT_CHANS')
    :param target_func: function to call, first arg must be channel number
    :param func_args: function argument tuple NOT INCLUDING channel number e.g. (sig_data, goodPixelsOneIsMask_RowCol)
    :param multithread: if True allocate a separate thread for each channel; otherwise the channels are processed sequentially in a single thread
        
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