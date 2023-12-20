'''
Computes the sums of normal and reference pixel combinations for a single ramp file

Specifically, given the following definitions

    n -> normal pixel
    a -> reference output
    rl -> left reference pixels
    r1 -> right reference pixels

compute the following:

    n*n
    n*a
    n*rl
    n*r1
    a*a
    a*rl
    a*r1
    rl*rl
    r1*r1
    rl*r1 

@author: rarendt
@author: smaher
'''

import os
import time

from astropy import stats
import h5py
import numpy as np
import scipy.fft as spfft
from irrc_constants import NUM_OUTPUT_CHANS, END_OF_ROW_PIXEL_PAD, \
    NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD, NUM_COLS_PER_OUTPUT_CHAN, NUM_ROWS, \
    NUM_COLS, REFPIX_NORM, PIXEL_READ_FREQ_HZ


## TO IMPORT STILL 
from irrc import util 
# from irrc.config import cfgOutlierStdDevThreshold, cfgFFTInterpolationIterations
from irrc.util import read_roman_fits, exec_channel_func_threads

## GET RID OF
# from irrc import logger
# Allocate singleton logger for this module
# logger = logger.get_logger(__name__)


#
# Pixel values with sigmas larger than this value are removed (and interpolated over) 
# during ramp sum extraction.  

# NOTE: this is for the default outlier function; a custom outlier function may ignore this.
cfgOutlierStdDevThreshold = 4.0

# Number of iterations when doing FFT interpolations
#
cfgFFTInterpolationIterations = 3

def extract(inFileName:str, outDirectory:str=None, multiThread:bool=True, 
    skipFirstFrame:bool=True, externalPixelFlags:np.ndarray=None, externalOutlierFunc=None, outlierStdDev:float=cfgOutlierStdDevThreshold):
    '''
    Extract ramp sums from a single ASDF file.  Generally this is done for several files and then generate is run on the results.
    
    :param inFileName: input FITS file name
    :param outDirectory: directory in which to store results
    :param multiThread: should multithreading be used in various calculations?
    :param skipFirstFrame: should the first frame of the data be skipped? (it is often skipped to avoid reset settling artifacts)
    :param externalPixelFlags: optional external pixel flags that get combined with internal outlier mask (as defined in applyExternalPixelFlagsToOutlierMask(); shape (constants.NUM_ROWS, constants.NUM_COLS) 
    :param externalOutlierFunc: optional alternative to default outlier function; same method signature as _find_outliers_chan_func
    :param outlierStdDev: number of standard deviations to be considered an 'outlier'.  Used by default outlier function and passed to custom outlier func
    '''

    startSec = time.time()
    
    if not os.path.exists(inFileName):
        mesg = f'Input file {inFileName} does not exist. Terminating.'
        # logger.fatal(mesg)
        raise FileNotFoundError(mesg)
        
    print(f'Performing ramp sum calculation on file {inFileName}')
    
    ext = "_sums.h5"
    if not outDirectory:
        outFileName = os.path.basename(inFileName) + ext
    else:
        if not os.path.exists(outDirectory):
            mesg = f'Output directory {outDirectory} does not exist. Terminating.'
            # logger.fatal(mesg)
            raise FileNotFoundError(mesg)
            
        outFileName = outDirectory + '/' + os.path.basename(inFileName) + ext
    

            
    print(f'Input FITS inFileName: {inFileName}')
    print(f'Output inFileName: {outFileName}')
    
    
    
    ####################### CHANGE TO READ IN ASDF!!!! ##################
    ## FOR NOW, ASSUME THE DATA IS IN THE CORRECT FORMAT
    # Read FITS file
    data0 = read_roman_file(inFileName, skipFirstFrame, logger)
    numFrames = data0.shape[0]
    if numFrames < 2:
        # _sum_chan_func() requires > 1 frames with (frames - 1) as denominator
        # logger.fatal(f'IRRC does not support FITS files with fewer than two frames.  File {inFileName} has {numFrames} frames')
        raise ValueError("Illegal number of frames")
    
    
    #######################
    #
    # Optional pre-apply externalPixelFalgs to data
    #
    # If an external pixel flag array is provided, this is an optional hook use it to modify the incoming data
    if externalPixelFlags is not None:
        # logger.info('Applying externalPixelFlags to incoming data')
        preApplyExternalPixelFlagsToData(data0, externalPixelFlags)
      

    #######################
    #
    msg = 'Removing linear slopes and offsets'
    #    
    # logger.info(msg)
    util.remove_linear_trends(data0, False)
    
    
    
    #######################
    #
    msg = 'Generate outlier mask'
    #
    # logger.info(msg)
    # The mask is size of the full pixel field (NUM_ROWS, NUM_COLS).  A value of 0
    # means an outlier and 1 means an inlier.  Outlier pixels in the data will obtain interpolated 
    # values before the ramp sums are generated.
    #
    # The generating function can be overridden.  The default function classifies outliers as those
    # with a value > outliersStdDev * standard deviation.  NOTE: the reference pix
    
    if externalOutlierFunc == None:
        outliersMask_RowCol = np.ones((NUM_ROWS, NUM_COLS), dtype=bool)
        
        # Prepare one-sigma data for outlier flagging
        # (mean(data0) = 0 after linear trend removal)
        sig_data = np.sqrt(np.sum(data0 ** 2 / (numFrames - 1), axis=0))  
        
        # logger.info(f'Flagging pixels outside stdev = {cfgOutlierStdDevThreshold}')
        exec_channel_func_threads(range(NUM_OUTPUT_CHANS - 1), _find_outliers_chan_func, (util.getReferenceMask(0), 
            sig_data, outliersMask_RowCol, outlierStdDev), multiThread=multiThread)
            
    
        # expand outliersMask_RowCol to mask x,y neighbor of outlier pixels.
        outliersMask_RowCol = outliersMask_RowCol * np.roll(outliersMask_RowCol, (1, 0), (0, 1)) * np.roll(outliersMask_RowCol,
            (0, 1), (0, 1)) * np.roll(outliersMask_RowCol, (-1, 0), (0, 1)) * np.roll(outliersMask_RowCol, (0, -1), (0, 1))
    else: 
        outliersMask_RowCol = externalOutlierFunc(data0, (NUM_ROWS, NUM_COLS), outlierStdDev=outlierStdDev)
    
    # reset reference output (last output channel) to not masked
    outliersMask_RowCol[:, -NUM_COLS_PER_OUTPUT_CHAN:] = 1
    
    
    #######################
    #
    # Optional modification of outlier mask by external data quality data
    #
    
    # If an external pixel flag array is provided, call a function to use it to modify the outlier mask.
    # It is expected the function will be modified once ROMAN pixel masks are defined
    if externalPixelFlags is not None:
        # logger.info('Applying externalPixelFlags to outlier mask')
        applyExternalPixelFlagsToOutlierMask(outliersMask_RowCol, externalPixelFlags)
        
    
    
    #######################
    #    
    msg = 'Apply outlier mask to data (i.e., set outlier pixels to 0)'
    #
    # logger.info(msg)
    for frameNum in range (numFrames):
        data0[frameNum,:,:] *= outliersMask_RowCol
    
    
    #######################
    #    
    msg = 'Convert data and mask from pixel space to time domain by padding'
    #
    # logger.info(msg)
    # From ([frames], flattenedFrame) to (allOutputChans, [frames], rows, cols) 
    outliersMask_ChanRowCol = np.transpose(outliersMask_RowCol.reshape((NUM_ROWS, NUM_OUTPUT_CHANS, NUM_COLS_PER_OUTPUT_CHAN)), (1, 0, 2))
    data_chansFramesRowsPhyscols = np.transpose(data0.reshape((numFrames, NUM_ROWS, NUM_OUTPUT_CHANS, NUM_COLS_PER_OUTPUT_CHAN)), (2, 0, 1, 3))

    # logger.info('... Undo alternating reversed readout order to put pixels in time order')
    for chan in range(1, NUM_OUTPUT_CHANS, 2):
        data_chansFramesRowsPhyscols[chan,:,:,:] = data_chansFramesRowsPhyscols[chan,:,:,::-1]
        outliersMask_ChanRowCol[chan,:,:] = outliersMask_ChanRowCol[chan,: ,::-1]
    
    # logger.info('... Add pad to rows to introduce appropriate delay from guide window scan, etc (i.e., create uniform sample timing)')
    dataUniformTime = np.pad(data_chansFramesRowsPhyscols, ((0, 0), (0, 0), (0, 0), (0, END_OF_ROW_PIXEL_PAD)))
    
    
    
    #######################
    #    
    msg = 'Remove linear trends at frame boundary'
    #
    # logger.info(msg)
    util.remove_linear_trends_per_frame(logger, dataUniformTime, subtractOffsetOnly=False, multiThread=multiThread)
            


    #######################
    #    
    # Cosine interpolation
    #
    
    # logger.info('Perform cosine weighted interpolation on zero values to provide preliminary values for bad pixels')
    
    # dataUniformTime has zero values as a result of 1) earlier flagged outlier pixels, 2) padding for uniform timing,
    # and 3) original 0's in the raw data (from broken pixel, etc.).  The interp_zeros_channel_fun applies the interpolation
    # to all zero values in order to smooth discontinuities before the FFT interpolation 
    # w = np.sin(np.arange(NUM_OUTPUT_CHANS, dtype=np.float64) / 32 * np.pi)[1:]
    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), util.interp_zeros_channel_fun,
        (util.getTrigInterpolationFunction(dataUniformTime), dataUniformTime, numFrames, NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD), 
        multiThread=multiThread)



    #######################
    #    
    # Prepare reference data (while data is in convenient form)
    #
    
    # logger.info('Prepare left and right column reference pixels') 
    
    
    rl = np.copy(dataUniformTime[0,:,:,:])  # need to copy otherwise a reference!
    rr = np.copy(dataUniformTime[31,:,:,:])
    # zero reference columns (data is in time domain)
    rl[:,:, 4:] = 0.
    rr[:,:, 4:] = 0.
    
    rl = np.reshape(rl, (numFrames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS))
    rr = np.reshape(rr, (numFrames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS))

    rlFFT = spfft.rfft(rl / rl[0].size) * REFPIX_NORM
    rrFFT = spfft.rfft(rr / rr[0].size) * REFPIX_NORM
              
    

    #######################
    #    
    # FFT interpolation
    #
    
    # logger.info('Perform FFT interpolation')
    
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # ; Use Fourier filter/interpolation to replace
    # ; (a) bad pixel, gaps, and reference data in the time-ordered normal data
    # ; (b) gaps and normal data in the time-ordered reference data
    # ; This "improves" upon the cosine interpolation performed above.
 
    dataUniformTime = np.reshape(dataUniformTime, (NUM_OUTPUT_CHANS, numFrames, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS)).astype(dtype=np.float64)

    # rfft returns size n//2 + 1 
    dataFFTOut = np.zeros((dataUniformTime.shape[0], dataUniformTime.shape[1], dataUniformTime.shape[2] // 2 + 1), dtype=np.complex128)
    
    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), util.fft_interp_step_channel_fun, 
        (dataUniformTime, dataFFTOut, outliersMask_ChanRowCol, util.getFFTApodizeFunction(), cfgFFTInterpolationIterations), multiThread=multiThread)
    
    
    #######################
    #    
    # Optionally apply external pixel flags to interpolated data
    # 
        
    # If an external pixel flag array is provided, this is an optional hook use it to modify the interpolated (and FFT'd) data
    if externalPixelFlags is not None:
        # logger.info('Applying externalPixelFlags to outlier mask')
        applyExternalPixelFlagsToInterpolatedData(dataFFTOut, externalPixelFlags)
            

    #######################
    #    
    # Calculate sums
    # 
    
    # logger.info('Sum calculation ..')
    sum_nn = np.sum(np.square(np.abs(dataFFTOut)), 1) / (numFrames - 1)
    sum_na = sum_nn.astype(complex)
    sum_nl = np.copy(sum_na)
    sum_nr = np.copy(sum_na)
    
    conjData = np.conjugate(dataFFTOut[-1,:,:])
    conjrl = np.conjugate(rlFFT)
    conjrr = np.conjugate(rrFFT)            

    exec_channel_func_threads(range(NUM_OUTPUT_CHANS), _sum_chan_func, (dataFFTOut, sum_na, sum_nl, sum_nr, conjData, conjrl, conjrr, numFrames), multiThread=multiThread)
    
    sum_ll = np.sum(np.abs(rlFFT) ** 2, 0) / (numFrames - 1)
    sum_rr = np.sum(np.abs(rrFFT) ** 2, 0) / (numFrames - 1)  # total(abs(r1)^2,2)/(s[3]-1)
    sum_lr = np.sum(rlFFT * conjrr, 0) / (numFrames - 1)  # total(rl*conj(r1),2)/(s[3]-1)



    #######################
    #    
    # Write results
    # 
        
    e = NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * NUM_ROWS
    f = np.abs(np.fft.rfftfreq(e, 1 / e))
    freq = f * (PIXEL_READ_FREQ_HZ / 2.) / f.max()
    
    # logger.info(f'Writing to {outFileName}')
    with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset("freq", data=freq)
        hf.create_dataset("sum_nn", data=sum_nn)
        hf.create_dataset("sum_na", data=sum_na)
        hf.create_dataset("sum_nl", data=sum_nl)
        hf.create_dataset("sum_nr", data=sum_nr)
        hf.create_dataset("sum_ll", data=sum_ll)
        hf.create_dataset("sum_rr", data=sum_rr)
        hf.create_dataset("sum_lr", data=sum_lr)
    # logger.info(f'Total wall clock execution (seconds):  {time.time() - startSec}')
    # logger.info('Done')
    
    return (sum_nn, sum_na, sum_nl, sum_nr, sum_ll, sum_rr, sum_lr)




def _find_outliers_chan_func(chan:int, refZeroMask:np.ndarray, sig_data:np.ndarray, outliersMask_RowCol_InOut:np.ndarray, outlierStdDev:float=cfgOutlierStdDevThreshold):
    '''
    Set outliersMask_RowCol_InOut to 0 where the sig_data values are >= std of sigmaThreshold.  The evaluation is done separately for
    reference and normal pixels.
    :param chan: channel number
    :param refZeroMask: IN ONLY full image mask where reference pixels have value 0
    :param sig_data:
    :param outliersMask_RowCol_InOut:  OUT ONLY Sets mask 0 = where pixel value is greater than std dev threshold) 
    :param outlierStdDev: 
    '''

    columnSlice = slice(chan * NUM_COLS_PER_OUTPUT_CHAN, (chan + 1) * NUM_COLS_PER_OUTPUT_CHAN)
    columnData = (sig_data[:, columnSlice] - sig_data[:, -NUM_COLS_PER_OUTPUT_CHAN:])
    
    # Python masked arrays use a 1 where the mask exists (invalid values), so the refZeroMask allows 
    refZeroMaskChan = refZeroMask[:, columnSlice]
    
    # Python masked arrays use a 1 where the mask exists (e.g. 1 = ignore values). Therefore refZeroMaskChan, with 0 = reference pixel
    #  will have sigma_clip operate on just the reference pixels

    refPixsInliers = stats.sigma_clip(np.ma.array(columnData, mask=refZeroMaskChan), cenfunc='median', stdfunc='std', sigma=outlierStdDev, masked=True)
    
    # The refPixsInliers mask will have 0's for inliers.  Change that to 1's for inliers
    refPixsInliersOneMask = np.invert(refPixsInliers.mask)
            
    # Invert refZeroMaskChan to now have sigma_clip operated on normal pixels
    normalZeroMask = np.invert(refZeroMaskChan) 
    normPixInliers = stats.sigma_clip(np.ma.array(columnData, mask=normalZeroMask), cenfunc='median', stdfunc='std', sigma=outlierStdDev, masked=True)
    normPixsInliersOneMask = np.invert(normPixInliers.mask)

    # Combine ref and normal pixel inlier (val = 1) masks so the return array has 0's where 
    outliersMask_RowCol_InOut[:, columnSlice] = np.ma.mask_or(normPixsInliersOneMask, refPixsInliersOneMask)
    

def _sum_chan_func(chan:int, dataFFTOut:np.ndarray, sum_no:np.ndarray, sum_nrl:np.ndarray, sum_nr1:np.ndarray,
                  conjd:np.ndarray, conjrl:np.ndarray, conjr1:np.ndarray, numFrames:int):
    
    frameData = dataFFTOut[chan,:,:]
    sum_no[chan,:] = np.sum(frameData * conjd, 0) / (numFrames - 1)
    sum_nrl[chan,:] = np.sum(frameData * conjrl, 0) / (numFrames - 1)
    sum_nr1[chan,:] = np.sum(frameData * conjr1, 0) / (numFrames - 1)      
    



#################################################
#
# Function hooks for applying external pixel quality flags to various parts of the processing
#

def preApplyExternalPixelFlagsToData(data:np.ndarray, externalPixelFlags:np.ndarray):
    '''
    Called after reading input FITS file.
    
    Optionally use external flags to directly change the detector data after reading from FITS file.
    This could be used to artifically set pixels to values that would be flagged by the outlier function.
    
    An eguivalent operation would be to provide an externalOutlierFunc to extract() and explictly generate
    the outlier mask.
    :param data: IN/OUT detector data
    :param externalPixelFlags: IN external pixel/quality flags provided to extract()
    '''

def applyExternalPixelFlagsToOutlierMask(outlierMask_RowCol:np.ndarray, externalPixelFlags:np.ndarray):
    '''
    Called after outlier mask is generated.
    
    Combine internal outlier mask with external pixel flags.
        
    :param outlierMask_RowCol: IN/OUT shape (constants.NUM_ROWS, constants.NUM_COLS) where 0 indicates outlier pixels that will have their values interpolated
    :param externalPixelFlags: IN shape (constants.NUM_ROWS, constants.NUM_COLS) incoming pixel flag array
    '''
    # Goofy example for unit testing ... 
    outlierMask_RowCol[(externalPixelFlags == 0) | (externalPixelFlags <= -2)] = 0

def applyExternalPixelFlagsToInterpolatedData(data_RowCol:np.ndarray, externalPixelFlags:np.ndarray):
    '''
    Called after data interpolation but before sum calculation
    
    Apply external pixel flags to interpolated data
        
    :param outlierMask_RowCol: shape (constants.NUM_ROWS, constants.NUM_COLS) where 0 indicates outlier pixels that will have their values interpolated
    :param externalPixelFlags: shape (constants.NUM_ROWS, constants.NUM_COLS) incoming pixel flag array
    '''
    pass