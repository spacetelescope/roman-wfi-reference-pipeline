'''
Various functions that are common to multiple IRRC steps

@author: smaher
@author: rarendt
'''

import threading
import unittest

from astropy.io import fits

import numpy as np
import scipy.fft as spfft
import h5py

from .irrc_polynomial import Polynomial
from .irrc_constants import NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD, NUM_OUTPUT_CHANS, \
    ALL_CHAN_RANGE, REFERENCE_ROWS, FITS_DATA_HDU_NUMBER, \
    NUM_COLS, NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN

import logging
logger = logging.getLogger('ReferencePixel util')

def read_roman_file(fileName:str, skipFirstFrame:bool, logger:logging.Logger):
    if '.h5' in fileName:
        return read_roman_hdf5(fileName, skipFirstFrame, logger)
    elif '.fits' in fileName:
        return read_roman_fits(fileName, skipFirstFrame, logger)
    elif '.asdf' in fileName:
        return read_roman_asdf(fileName, skipFirstFrame, logger)
    else:
        raise ValueError('can only read in .h5 and .fits.  working on asdf now. ')

def read_roman_fits(fileName:str, skipFirstFrame:bool, logger:logging.Logger):
    '''
    Read a ROMAN FITS file
    :param fileName: FITS file name
    :param skipFirstFrame: optionally skip first frame
    :param logger: e.g., irrc.logger.get_logger(name)
    :returns FITS image array
    '''


    logger.info(f"Trying to read FITS file: {fileName}")
    # Cannot use a memory-mapped image: BZERO/BSCALE/BLANK header keywords present.
    with fits.open(fileName, memmap=False) as hduList:

        hduData = hduList[FITS_DATA_HDU_NUMBER]
        logger.info(f"Reading FITS data of dimensions {hduData.header['NAXIS1']}  x  {hduData.header['NAXIS2']} x {hduData.header['NAXIS3']}")

        # numpy.ndarray of type uint16 then float64 after multiplication by gain
        # FITS has an extra dimension, hence the data[0]
        # Also remove the first frame due to reset anomalies
        if skipFirstFrame:
            data0 = hduData.data[0, 1:]
        else:
            data0 = hduData.data[0,:]

    dataShape = data0.shape
    numCols = dataShape[2]
    numRows = dataShape[1]

    if numCols != NUM_COLS or numRows != NUM_ROWS:
        raise Exception("File:", fileName, " has incorrect dimensions.  Expecting numRows =", NUM_ROWS, ", numCols =", NUM_COLS)

    # Convert from uint16 to prepare for in-place computations
    return data0.astype(np.float64)

def read_roman_hdf5(fileName:str, skipFirstFrame:bool, logger:logging.Logger):
    '''
    read a ROMAN hdf5 file
    written by Sarah Betti based on read_roman_fits()
    returns a image array
    '''
    logger.info(f"Trying to read HDF5 file: {fileName}")

    fil = h5py.File(fileName, 'r')
    # get data
    dset = fil['Frames']
    if skipFirstFrame:
        data0 = np.array(list(dset))[1:]
    else:
        data0 = np.array(list(dset))

    dataShape = data0.shape
    numCols = dataShape[2]
    numRows = dataShape[1]

    if numCols != NUM_COLS or numRows != NUM_ROWS:
        raise Exception("File:", fileName, " has incorrect dimensions.  Expecting numRows =", NUM_ROWS, ", numCols =", NUM_COLS)
    
    # Convert from uint16 to prepare for in-place computations
    return data0.astype(np.float64)

def read_roman_asdf(fileName:str, skipFirstFrame:bool, logger:logging.Logger):
    raise ValueError('cannot read in ASDF yet! ')



def write_slopes(dataFramesRowsCols:np.ndarray, fileName:str):
    '''
    Generate diagnostic per-pixel fitted slopes image and write to FITS file 
    :param dataFramesRowsCols:
    :param fileName:
    '''
    sh = dataFramesRowsCols.shape
    nz = sh[0]
    ny = sh[1]
    nx = sh[2]
    P = Polynomial(nz, 1)
    S = np.empty((1,ny,nx))
    S[0] = P.polyfit(dataFramesRowsCols)[1]
    fits.PrimaryHDU(S).writeto(fileName, overwrite=True)
    logger.info(f'Wrote slopes to file {fileName}')

    
def remove_linear_trends(data_FramesRowsCols:np.ndarray, subtractOffsetOnly:bool):
    '''
    Remove linear trends (gains/slope, biases/offsets) per pixel 
    :param data_FramesRowsCols: image data [numFrames][numRows][numCols].  The data is modified IN PLACE
    :param subtractOffsetOnly: if True, only the liner model offset is removed.  If False, the offset and slope are removed
    :return per-pixel image array of modeled m (slope) and b (offset) 
    '''

    numFrames = len(data_FramesRowsCols)
    numRows = len(data_FramesRowsCols[0])
    numCols = len(data_FramesRowsCols[0][0])

    # for mat multiply
    data_FramesRowsCols = data_FramesRowsCols.reshape((numFrames, numCols * numRows))

    frameRamp = np.arange(numFrames, dtype=data_FramesRowsCols.dtype)

    # subtract mean(frameRampMinusMean) to minimize correlation of slope and offset
    frameRampMinusMean = frameRamp - np.mean(frameRamp)

    sxy = np.matmul(frameRampMinusMean, data_FramesRowsCols, dtype=data_FramesRowsCols.dtype)
    sx = np.sum(frameRampMinusMean)
    sxx = np.sum(frameRampMinusMean ** 2)
    sy = np.sum(data_FramesRowsCols, axis=0)

    m = (numFrames * sxy - sx * sy) / (numFrames * sxx - sx ** 2)
    b = (sy * sxx - sxy * sx) / (numFrames * sxx - sx ** 2)

    for frame in range(numFrames):
        if not subtractOffsetOnly:
            data_FramesRowsCols[frame,:] -= frameRampMinusMean[frame] * m + b  # subtract slope and offset
        else:
            data_FramesRowsCols[frame,:] -= b  # subtract offset

    data_FramesRowsCols = data_FramesRowsCols.reshape(numFrames, numRows, numCols)
    return m.reshape(numRows, numCols), b.reshape(numRows, numCols)


def remove_linear_trends_per_frame(logger:logging.Logger, data_chansFramesRowsChancols:np.ndarray, subtractOffsetOnly:bool, multiThread=True):
    '''
    Entry point for Fitting and removal of slopes per frame to remove issues at frame boundaries. 
    :param data_chansFramesRowsChancols: [numOutChannels(33)][numFrames][numRows][numColsWithPadding].  The data is modified IN PLACE
    :param subtractOffsetOnly: if True, only the liner model offset is removed.  If False, the offset and slope are removed
    '''

    numFrames = len(data_chansFramesRowsChancols[0])
    numRows = len(data_chansFramesRowsChancols[0][0])
    timeDomainRange = np.arange(NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD * numRows, dtype=data_chansFramesRowsChancols.dtype).reshape(numRows, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD)
    timeDomainRange = timeDomainRange - timeDomainRange.mean()

    exec_channel_func_threads(ALL_CHAN_RANGE, remove_linear_trends_per_frame_chan_func,
            (logger, numFrames, timeDomainRange, data_chansFramesRowsChancols, subtractOffsetOnly), multiThread=multiThread)


def remove_linear_trends_per_frame_chan_func(chanNumber:int, logger:logging.Logger, numFrames:int,
    timeDomainRange:np.ndarray, data_chansFramesRowsChancols:np.ndarray, subtractOffsetOnly:bool):
    '''
    Per-channel function for fitting and removal of slopes per frame to remove issues at frame boundaries - single channel only
    :param chanNumber:
    :param numFrames:
    :param timeDomainRange:
    :param data_chansFramesRowsChancols: [numOutChannels(33)][numFrames][numRows][numColsWithPadding].  The data is modified IN PLACE
    :param subtractOffsetOnly: if True, only the liner model offset is removed.  If False, the offset and slope are removed
    '''

    ab_3 = np.zeros((NUM_OUTPUT_CHANS, numFrames, 2), dtype=np.float64)  # for top+bottom ref pixel rows
    tdRef = timeDomainRange[REFERENCE_ROWS,:]

    for frame in range(numFrames):
        xVals = tdRef[(data_chansFramesRowsChancols[chanNumber, frame, REFERENCE_ROWS,:] != 0)]  # where data[chanNumber,frame,row4plus4,:] != 0)
        data2 = data_chansFramesRowsChancols[chanNumber, frame, REFERENCE_ROWS,:]
        yVals = data2[(data2 != 0)]
        if xVals.size < 1 or yVals.size < 1:
            logger.warn(f'Skipping empty data section.  Frame={frame}, Chan={chanNumber}')
            continue

        ab_3[chanNumber, frame,:] = np.polyfit(xVals, yVals, 1)

        dataNotZero = (data_chansFramesRowsChancols[chanNumber, frame,:,:] != 0)

        if not subtractOffsetOnly:
            data2 = (timeDomainRange * ab_3[chanNumber, frame, 0] + ab_3[chanNumber, frame, 1])
        else:
            data2 = ab_3[chanNumber, frame, 1]

        data_chansFramesRowsChancols[chanNumber, frame,:,:] -= data2 * dataNotZero


def interp_zeros_channel_fun(chanNumber:int, interpFunc:np.ndarray, data_ChansFramesRowsCols:np.ndarray, numFrames:int, numRows:int, numColsPerOutputChanWithPad:int):
    '''
    Convolve interpFunc across pixels with values of 0 for a single channel.  We are not using the outlier mask and thus possibly interpolating
    pixels with original values of 0.  This is to ensure we don't have discontinuities (around 0) when doing the FFT interpolation, which could
    damage the performance.
    
    :param chanNumber:
    :param interpFunc:
    :param data_ChansFramesRowsCols:  Data to interpolate, which is done IN PLACE
    :param numFrames:
    :param numRows:
    :param numColsPerOutputChanWithPad:
    '''
    for frame in range(numFrames):
        dat = np.reshape(data_ChansFramesRowsCols[chanNumber, frame,:,:], numColsPerOutputChanWithPad * numRows)

        # Apply to areas with 0 value.
        applyMask = np.where(dat != 0, 1, 0)

        dataConvolve = np.convolve(dat, interpFunc, mode='same')  # 'same' -> /edge_wrap in IDL
        applyMaskConvolve = np.convolve(applyMask, interpFunc, mode='same')

        sfm3 = dataConvolve / applyMaskConvolve
        dat = np.reshape(sfm3, (numRows, numColsPerOutputChanWithPad))
        mask2D = np.reshape(applyMask, (numRows, numColsPerOutputChanWithPad))
        data_ChansFramesRowsCols[chanNumber, frame,:,:] += dat * (1 - mask2D)
        # Repair any NaNs
        np.nan_to_num(data_ChansFramesRowsCols[chanNumber,:,:,:], copy=False)


def fft_interp_step_channel_fun(chanNumber:int, data_chansFramesFlattenedimage:np.ndarray, dataFFTOut:np.ndarray, readOnlyPixelsAreOneMask_ChanRowCol:np.ndarray, appodize_func:np.ndarray, numFFTIterations:int):
    '''
    Perform FFT interpolation on pixels for a single channel. A final FFT is done before return (needed by IRRC algorithm and more performant to do here) 
    :param chanNumber:
    :param data_chansFramesFlattenedimage: (channels, frames, rows*cols) of data.  WILL BE UPDATED IN PLACE
    :param dataFFTOut: Returned interpolated data - with a final FFT applied 
    :param readOnlyPixelsAreOneMask_ChanRowCol: image mask with 1's represent read-only pixels (e.g., 0 entries will be updated with interpolation)
    :param appodize_func:
    :param numFFTIterations:
    '''
    chanData_FramesFlat = data_chansFramesFlattenedimage[chanNumber,:,:]
    readOnlyPixelsAreOnesFrameMask_RowsColsPadded = np.zeros((NUM_ROWS, NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD), dtype=bool)
    readOnlyPixelsAreOnesFrameMask_RowsColsPadded[:,:NUM_COLS_PER_OUTPUT_CHAN] = readOnlyPixelsAreOneMask_ChanRowCol[chanNumber,:,:]

    fft_interp(chanData_FramesFlat, readOnlyPixelsAreOnesFrameMask_RowsColsPadded.flatten(), appodize_func, numFFTIterations)

    dataFFTOut[chanNumber,:,:] = spfft.rfft(chanData_FramesFlat / chanData_FramesFlat[0].size)


def fft_interp(chanData_FramesFlat:np.ndarray, readOnlyPixelsAreOneMask_Flattenedimage:np.ndarray, apodizeFilter:np.ndarray, numIterations:int):
    '''
    Perform FFT interpolation on pixels not marked as read-only
    :param chanData_FramesFlat: Multi-frame, single channel data with each frame flattened to 1D.  WILL BE UPDATED IN PLACE
    :param readOnlyPixelsAreOneMask_Flattenedimage:  Flattened image size mask indicating (good) pixels that should be used in the interpolation but not modified
    :param numFrames: number of frames in chanData_FramesFlat
    :param apodizeFilter:
    :param numIterations: number of times to iterate the interpolation
    '''

    numFrames = len(chanData_FramesFlat)
    readOnlyPixelsIndices_Flat = np.where(readOnlyPixelsAreOneMask_Flattenedimage)[0]

    for frame in range(numFrames):

        chanFrameData_Flat = chanData_FramesFlat[frame]
        readOnlyPixelsValues_Flat = chanFrameData_Flat[readOnlyPixelsIndices_Flat]

        for _ in range(numIterations):

            fftResult = apodizeFilter * spfft.rfft(chanFrameData_Flat, workers=1) / chanFrameData_Flat.size
            chanFrameData_Flat = spfft.irfft(fftResult * chanFrameData_Flat.size, workers=1)

            # Return read only pixels
            chanFrameData_Flat[readOnlyPixelsIndices_Flat] = readOnlyPixelsValues_Flat

        chanData_FramesFlat[frame,:] = chanFrameData_Flat


def getReferenceMask(valueForRefPixels:bool=0):
    '''
    Generate a full image mask of the reference pixels
    
    :param valueForRefPixels: set reference pixels to this value
    '''

    if valueForRefPixels == True:
        mask = np.zeros((NUM_ROWS, NUM_COLS), dtype=bool)
    else:
        mask = np.ones((NUM_ROWS, NUM_COLS), dtype=bool)

    # set reference pixels to 0
    mask[:, 0:4] = valueForRefPixels
    mask[:, 4092:4096] = valueForRefPixels
    mask[0:4, 0:4096] = valueForRefPixels
    mask[-4:, 0:4096] = valueForRefPixels
    return mask

def getFFTApodizeFunction():
    
    apo_len = NUM_ROWS * NUM_COLS_PER_OUTPUT_CHAN_WITH_PAD
    apo2 = np.abs(np.fft.rfftfreq(apo_len, 1 / apo_len))
    apodize_func = np.cos(2 * np.pi * apo2 / apo_len) / 2.0 + 0.5 
    return apodize_func

def getTrigInterpolationFunction(data):
    return np.sin(np.arange(NUM_OUTPUT_CHANS, dtype=data.dtype) / 32 * np.pi)[1:]

def exec_channel_func_threads(chanIndexRange:range, targetFunc, funcArgs, multiThread=False):
    '''
    Execute a function over a range of channels.  If multiThread is True, all executions
    will be executed on individual threads.  
    
    :param chanIndexRange: E.g., 'range(NUM_OUTPUT_CHANS')
    :param targetFunc: function to call, first arg must be channel number
    :param funcArgs: function argument tuple NOT INCLUDING channel number e.g. (sig_data, goodPixelsOneIsMask_RowCol)
    :param multiThread: if True allocate a separate thread for each channel; otherwise the channels are processed sequentially in a single thread
        
    '''

    if multiThread:
        logger.info("Multithreading-> ")
        threadList = []
        for c in chanIndexRange:
            funcArgsWithChan = (c,) + funcArgs
            threadList.append(threading.Thread(target=targetFunc, args=funcArgsWithChan))

        for t in threadList:
            t.start()
            # print("m", end='', flush=True)  # 'm' -> multi-threading

        for t in threadList:
            t.join()
            # print("-", end='', flush=True)
    else:
        logger.info("Single threading-> ")
        for c in chanIndexRange:
            funcArgsWithChan = (c,) + funcArgs
            # print("s", end='', flush=True)  # 's' -> single threading
            targetFunc(*funcArgsWithChan)
            # print("-", end='', flush=True)
    logger.info('Done executing over range of channels')
    # print()  # EOL

    
    
def destripe(D:np.ndarray):
    '''
    Remove output offsets using inner 2 rows of reference columns on bottom and top of 
    image area. For each output, the reference correction is linearly interpolated between
    the reference rows.
    
    :param D: A Roman WFI datacube

    '''
    
    # Definitions
    ny,nx = 4096,4224  # Image dimensions
    nout = 33          # RST uses 32 outputs plus 1x reference output
    wout = 128         # 128 pixels per output
    count = 2          # Discard this many low/high samples for robust statistics
    y0,y1 = 1.5,4093.5 # We use the inner 2 reference rows on the bottom and top.
                       #   these are the mean y-coordinates for both.
    y = np.arange(4096, dtype=np.float32).reshape((1,1,-1,1)) # y-coordinates of all rows shaped for broadcasting
    
    # Means of reference pixels in rows. The inner two rows on the bottom
    # and top are most representative. We discard `count` samples on either 
    # end of the distribution to make it robust against outliers.
    B = np.mean(np.sort(D[:,1:3,:].reshape((-1,2,nout,wout)).swapaxes(2,3).reshape((-1,2*wout,
                                nout)), axis=1)[:,count:-count,:], axis=1).reshape((-1,33,1,1))
    T = np.mean(np.sort(D[:,-3:-1,:].reshape((-1,2,nout,wout)).swapaxes(2,3).reshape((-1,2*wout,
                                    nout)), axis=1)[:,count:-count,:], axis=1).reshape((-1,33,1,1))
    
    # Compute reference corrections for all outputs simultaneously interpolating between bottom and top
    R = B + (y-y0)*(T-B)/(y1-y0)
    
    # Subtract reference correction
    D = (D.reshape((-1,ny,nout,wout)).swapaxes(1,2) - R).swapaxes(1,2).reshape((-1,ny,nx))
    
    # Done!
    return(D)