'''
Computes the sums of normal and reference pixel combinations for a single ramp file

Specifically, given the following definitions

    n -> normal pixel
    a -> reference output
    rl -> left reference pixels
    rr -> right reference pixels

compute the following:
    n x n* 
    n x l*
    n x r*
    l x l*
    r x r*
    l x r*
    a x a*
    a x l*
    a x r*

Written by (Rauscher et al., in prep):
    - R. Arendt
    - S. Maher
'''

import logging
import time

import h5py
import numpy as np
import scipy.fft as spfft
from astropy import stats

from . import irrc_util as util
from .irrc_constants import (
    END_OF_ROW_PIXEL_PAD,
    NUM_COLS,
    NUM_COLS_PER_OUTPUT_CHAN,
    NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD,
    NUM_OUTPUT_CHANS,
    NUM_ROWS,
    PIXEL_READ_FREQ_HZ,
    REFPIX_NORM,
)
from .irrc_util import exec_channel_func_threads

logger = logging.getLogger('ReferencePixel Sums')


def extract(data:np.ndarray, out_file_name:str, multithread:bool=True, 
            external_pixel_flags:np.ndarray=None, external_outlier_func=None, outlier_stddev:float=4.0, cfg_fft_interpolation_iterations:int=3):
    '''
    Extract ramp sums from a single ASDF file.  Generally this is done for several files and then generate is run on the results.
    
    Parameters
    ----------
    data: ndarray
        input data
    out_file_name: str
        full file path in which to store results
    multithread: boolean, default = True
        should multithreading be used in various calculations?
    external_pixel_flags: npdarray, default = None
        optional external pixel flags that get combined with internal outlier mask (as defined in apply_external_pixel_flags_to_outlier_mask(); shape (constants.NUM_ROWS, constants.NUM_COLS) 
    external_outlier_func: default = None
        optional alternative to default outlier function; same method signature as _find_outliers_chan_func
    outlier_stddev: float, default = 4.0
        number of standard deviations to be considered an 'outlier'.  Used by default outlier function and passed to custom outlier func
    cfg_fft_interpolation_iterations: int, default = 3
        Number of iterations when doing FFT interpolations

    Returns
    ----------
    (sum_nn, sum_na, sum_nl, sum_nr, sum_ll, sum_rr, sum_lr): 
        list of sums of normal and reference pixel combinations
    '''

    start_sec = time.time()
    
    num_frames = data.shape[0]
    if num_frames < 2:
        logger.fatal(f'IRRC does not support exposures with fewer than two frames.  Data has {num_frames} frames')
        raise ValueError("Illegal number of frames")
    
    # Optional pre-apply externalPixelFalgs to data
    # If an external pixel flag array is provided, this is an optional hook use it to modify the incoming data
    if external_pixel_flags is not None:
        logger.info('Applying external_pixel_flags to incoming data')
        pre_apply_external_pixel_flags_to_data(data, external_pixel_flags)
      
    msg = 'Removing linear slopes and offsets'  
    logger.info(msg)
    util.remove_linear_trends(data, False)
    

    msg = 'Generate outlier mask'
    logger.info(msg)
    # The mask is size of the full pixel field (NUM_ROWS, NUM_COLS).  A value of 0
    # means an outlier and 1 means an inlier.  Outlier pixels in the data will obtain interpolated 
    # values before the ramp sums are generated.
    #
    # The generating function can be overridden.  The default function classifies outliers as those
    # with a value > outliersStdDev * standard deviation.  NOTE: the reference pix
    
    if external_outlier_func is None:
        outliers_mask_rowcol = np.ones((NUM_ROWS, NUM_COLS), dtype=bool)
        
        # Prepare one-sigma data for outlier flagging
        # (mean(data0) = 0 after linear trend removal)
        sig_data = np.sqrt(np.sum(data ** 2 / (num_frames - 1), axis=0))  
        
        logger.info(f'Flagging pixels outside stdev = {outlier_stddev}')
        exec_channel_func_threads(range(NUM_OUTPUT_CHANS - 1), _find_outliers_chan_func, (util.get_reference_mask(0), 
            sig_data, outliers_mask_rowcol, outlier_stddev), multithread=multithread)
            
    
        # expand outliers_mask_rowcol to mask x,y neighbor of outlier pixels.
        outliers_mask_rowcol = outliers_mask_rowcol * np.roll(outliers_mask_rowcol, (1, 0), (0, 1)) * np.roll(outliers_mask_rowcol,
            (0, 1), (0, 1)) * np.roll(outliers_mask_rowcol, (-1, 0), (0, 1)) * np.roll(outliers_mask_rowcol, (0, -1), (0, 1))
    else: 
        outliers_mask_rowcol = external_outlier_func(data, (NUM_ROWS, NUM_COLS), outlier_stddev=outlier_stddev)
    
    # reset reference output (last output channel) to not masked
    outliers_mask_rowcol[:, -NUM_COLS_PER_OUTPUT_CHAN:] = 1
    
    
    # Optional modification of outlier mask by external data quality data
    # If an external pixel flag array is provided, call a function to use it to modify the outlier mask.
    # It is expected the function will be modified once ROMAN pixel masks are defined
    if external_pixel_flags is not None:
        logger.info('Applying external_pixel_flags to outlier mask')
        apply_external_pixel_flags_to_outlier_mask(outliers_mask_rowcol, external_pixel_flags)
        
       
    msg = 'Apply outlier mask to data (i.e., set outlier pixels to 0)'
    logger.info(msg)
    for framenum in range (num_frames):
        data[framenum,:,:] *= outliers_mask_rowcol
       
    msg = 'Convert data and mask from pixel space to time domain by padding'
    logger.info(msg)
    # From ([frames], flattenedFrame) to (allOutputChans, [frames], rows, cols) 
    outliers_mask_chanrowcol = np.transpose(outliers_mask_rowcol.reshape((NUM_ROWS, NUM_OUTPUT_CHANS, NUM_COLS_PER_OUTPUT_CHAN)), (1, 0, 2))
    data_chans_frames_rowsphyscols = np.transpose(data.reshape((num_frames, NUM_ROWS, NUM_OUTPUT_CHANS, NUM_COLS_PER_OUTPUT_CHAN)), (2, 0, 1, 3))

    logger.info('... Undo alternating reversed readout order to put pixels in time order')
    for chan in range(1, NUM_OUTPUT_CHANS, 2):
        data_chans_frames_rowsphyscols[chan,:,:,:] = data_chans_frames_rowsphyscols[chan,:,:,::-1]
        outliers_mask_chanrowcol[chan,:,:] = outliers_mask_chanrowcol[chan,: ,::-1]
    
    logger.info('... Add pad to rows to introduce appropriate delay from guide window scan, etc (i.e., create uniform sample timing)')
    data_uniform_time = np.pad(data_chans_frames_rowsphyscols, ((0, 0), (0, 0), (0, 0), (0, END_OF_ROW_PIXEL_PAD)))
      
    msg = 'Remove linear trends at frame boundary'
    logger.info(msg)
    util.remove_linear_trends_per_frame(logger, data_uniform_time, subtract_offset_only=False, multithread=multithread)
    
    # Cosine interpolation
    logger.info('Perform cosine weighted interpolation on zero values to provide preliminary values for bad pixels')
    
    # data_uniform_time has zero values as a result of 1) earlier flagged outlier pixels, 2) padding for uniform timing,
    # and 3) original 0's in the raw data (from broken pixel, etc.).  The interp_zeros_channel_fun applies the interpolation
    # to all zero values in order to smooth discontinuities before the FFT interpolation 
    # w = np.sin(np.arange(NUM_OUTPUT_CHANS, dtype=np.float64) / 32 * np.pi)[1:]
    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), util.interp_zeros_channel_fun,
        (util.get_trig_interpolation_function(data_uniform_time), data_uniform_time, num_frames, NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD), 
        multithread=multithread)
  
    # Prepare reference data (while data is in convenient form)
    logger.info('Prepare left and right column reference pixels') 
    
    rl = np.copy(data_uniform_time[0,:,:,:])  # need to copy otherwise a reference!
    rr = np.copy(data_uniform_time[31,:,:,:])
    # zero reference columns (data is in time domain)
    rl[:,:, 4:] = 0.
    rr[:,:, 4:] = 0.
    
    rl = np.reshape(rl, (num_frames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS))
    rr = np.reshape(rr, (num_frames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS))

    rl_fft = spfft.rfft(rl / rl[0].size) * REFPIX_NORM
    rr_fft = spfft.rfft(rr / rr[0].size) * REFPIX_NORM
              
       
    # FFT interpolation
    logger.info('Perform FFT interpolation')
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # ; Use Fourier filter/interpolation to replace
    # ; (a) bad pixel, gaps, and reference data in the time-ordered normal data
    # ; (b) gaps and normal data in the time-ordered reference data
    # ; This "improves" upon the cosine interpolation performed above.
 
    data_uniform_time = np.reshape(data_uniform_time, (NUM_OUTPUT_CHANS, num_frames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS)).astype(dtype=np.float64)

    # rfft returns size n//2 + 1 
    data_fft_out = np.zeros((data_uniform_time.shape[0], data_uniform_time.shape[1], data_uniform_time.shape[2] // 2 + 1), dtype=np.complex128)
    
    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), util.fft_interp_step_channel_fun, 
        (data_uniform_time, data_fft_out, outliers_mask_chanrowcol, util.get_fft_apodize_function(), cfg_fft_interpolation_iterations), multithread=multithread)
      
    # Optionally apply external pixel flags to interpolated data  
    # If an external pixel flag array is provided, this is an optional hook use it to modify the interpolated (and FFT'd) data
    if external_pixel_flags is not None:
        logger.info('Applying external_pixel_flags to outlier mask')
        apply_external_pixel_flags_to_interpolated_data(data_fft_out, external_pixel_flags)
            
    # Calculate sums
    logger.info('Sum calculation ..')
    # sum_nn = sum_nl = sum_na = sum_nr > sums for the 33 amplifiers
    sum_nn = np.sum(np.square(np.abs(data_fft_out)), 1) / (num_frames - 1)
    sum_na = sum_nn.astype(complex)
    sum_nl = np.copy(sum_na)
    sum_nr = np.copy(sum_na)
    
    # conjugate of the 33rd reference amplifier
    conj_data = np.conjugate(data_fft_out[-1,:,:])
    # conjugate of the left reference columnes
    conjrl = np.conjugate(rl_fft)
    # conjugate of the right reference columns
    conjrr = np.conjugate(rr_fft)            

    # sum_na = (n x n*)
    # sum_nl = (n x l*)
    # sum_nr = (n x r*)
    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), _sum_chan_func, (data_fft_out, sum_na, sum_nl, sum_nr, conj_data, conjrl, conjrr, num_frames), multithread=multithread)
    
    # sum_ll = (l x l*)
    sum_ll = np.sum(np.abs(rl_fft) ** 2, 0) / (num_frames - 1)
    # sum_rr = (r x r*)
    sum_rr = np.sum(np.abs(rr_fft) ** 2, 0) / (num_frames - 1)  
    # sum_lf = (l x r*)
    sum_lr = np.sum(rl_fft * conjrr, 0) / (num_frames - 1)  

    # Write results
    e = NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS
    f = np.abs(np.fft.rfftfreq(e, 1 / e))
    freq = f * (PIXEL_READ_FREQ_HZ / 2.) / f.max()
    
    logger.info(f'Writing to {out_file_name}')
    with h5py.File(out_file_name, 'w') as hf:
        hf.create_dataset("freq", data=freq)
        hf.create_dataset("sum_na", data=sum_na) # (n x n*)
        hf.create_dataset("sum_nl", data=sum_nl) # (n x l*)
        hf.create_dataset("sum_nr", data=sum_nr) # (n x r*)
        hf.create_dataset("sum_ll", data=sum_ll) # (l x l*)
        hf.create_dataset("sum_rr", data=sum_rr) # (r x r*)
        hf.create_dataset("sum_lr", data=sum_lr) # (l x r*)
    logger.info(f'Total wall clock execution (seconds):  {time.time() - start_sec}')
    logger.info('Done')
    
    return (sum_nn, sum_na, sum_nl, sum_nr, sum_ll, sum_rr, sum_lr)


def _find_outliers_chan_func(chan:int, ref_zero_mask:np.ndarray, sig_data:np.ndarray, outliers_mask_rowcol_inout:np.ndarray, outlier_stddev:float=4.0):
    '''
    Set outliers_mask_rowcol_inout to 0 where the sig_data values are >= std of sigmaThreshold.  The evaluation is done separately for
    reference and normal pixels.

    Parameters
    ----------
    chan: int
        channel number
    ref_zero_mask: ndarray
        IN ONLY full image mask where reference pixels have value 0
    sig_data: ndarray
    outliers_mask_rowcol_inout: ndarray 
    OUT ONLY Sets mask 0 = where pixel value is greater than std dev threshold) 
    outlier_stddev: floaat
    '''

    column_slice = slice(chan * NUM_COLS_PER_OUTPUT_CHAN, (chan + 1) * NUM_COLS_PER_OUTPUT_CHAN)
    column_data = (sig_data[:, column_slice] - sig_data[:, -NUM_COLS_PER_OUTPUT_CHAN:])
    
    # Python masked arrays use a 1 where the mask exists (invalid values), so the ref_zero_mask allows 
    ref_zero_mask_chan = ref_zero_mask[:, column_slice]
    
    # Python masked arrays use a 1 where the mask exists (e.g. 1 = ignore values). Therefore ref_zero_mask_chan, with 0 = reference pixel
    #  will have sigma_clip operate on just the reference pixels

    ref_pixs_inliers = stats.sigma_clip(np.ma.array(column_data, mask=ref_zero_mask_chan), cenfunc='median', stdfunc='std', sigma=outlier_stddev, masked=True)
    
    # The ref_pixs_inliers mask will have 0's for inliers.  Change that to 1's for inliers
    ref_pixs_inliers_one_mask = np.invert(ref_pixs_inliers.mask)
            
    # Invert ref_zero_mask_chan to now have sigma_clip operated on normal pixels
    normal_zero_mask = np.invert(ref_zero_mask_chan) 
    norm_pix_inliers = stats.sigma_clip(np.ma.array(column_data, mask=normal_zero_mask), cenfunc='median', stdfunc='std', sigma=outlier_stddev, masked=True)
    norm_pixs_inliers_one_mask = np.invert(norm_pix_inliers.mask)

    # Combine ref and normal pixel inlier (val = 1) masks so the return array has 0's where 
    outliers_mask_rowcol_inout[:, column_slice] = np.ma.mask_or(norm_pixs_inliers_one_mask, ref_pixs_inliers_one_mask)
    

def _sum_chan_func(chan:int, data_fft_out:np.ndarray, sum_na:np.ndarray, sum_nl:np.ndarray, sum_nr:np.ndarray,
                  conj_data:np.ndarray, conjrl:np.ndarray, conjrr:np.ndarray, num_frames:int):
    
    frame_data = data_fft_out[chan,:,:]
    sum_na[chan,:] = np.sum(frame_data * conj_data, 0) / (num_frames - 1)
    sum_nl[chan,:] = np.sum(frame_data * conjrl, 0) / (num_frames - 1)
    sum_nr[chan,:] = np.sum(frame_data * conjrr, 0) / (num_frames - 1)      
    



#################################################
# Function hooks for applying external pixel quality flags to various parts of the processing

def pre_apply_external_pixel_flags_to_data(data:np.ndarray, external_pixel_flags:np.ndarray):
    '''
    Called after reading input file.
    
    Optionally use external flags to directly change the detector data after reading from  file.
    This could be used to artifically set pixels to values that would be flagged by the outlier function.
    
    An eguivalent operation would be to provide an external_outlier_func to extract() and explictly generate
    the outlier mask.

    Parameters
    ----------
    data: ndarray
        IN/OUT detector data
    external_pixel_flags: ndarray
        IN external pixel/quality flags provided to extract()
    '''
    pass

def apply_external_pixel_flags_to_outlier_mask(outlier_mask_rowcol:np.ndarray, external_pixel_flags:np.ndarray):
    '''
    Called after outlier mask is generated.
    
    Combine internal outlier mask with external pixel flags.

    Parameters
    ---------- 
    outlier_mask_rowcol: ndarray
        IN/OUT shape (constants.NUM_ROWS, constants.NUM_COLS) where 0 indicates outlier pixels that will have their values interpolated
    external_pixel_flags: ndarray
        IN shape (constants.NUM_ROWS, constants.NUM_COLS) incoming pixel flag array
    '''
    # Goofy example for unit testing ... 
    outlier_mask_rowcol[(external_pixel_flags == 0) | (external_pixel_flags <= -2)] = 0

def apply_external_pixel_flags_to_interpolated_data(data_rowcol:np.ndarray, external_pixel_flags:np.ndarray):
    '''
    Called after data interpolation but before sum calculation
    
    Apply external pixel flags to interpolated data
        
    Parameters
    ----------
    outlier_mask_rowcol: ndarray
        shape (constants.NUM_ROWS, constants.NUM_COLS) where 0 indicates outlier pixels that will have their values interpolated
    external_pixel_flags: ndarray
        shape (constants.NUM_ROWS, constants.NUM_COLS) incoming pixel flag array
    '''
    pass