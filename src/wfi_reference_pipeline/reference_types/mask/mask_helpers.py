import numpy as np

from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box2DKernel

import asdf
import roman_datamodels as rdm

def create_image_stats(data, sigma):
    """
    Calculating the sigma-clipped mean and stdev images from a stack of inputted images.
    """
    clipped_cube = sigma_clip(data,
                              sigma=sigma,
                              axis=0,
                              masked=False)

    mean_image = np.nanmean(clipped_cube,
                            axis=0)

    stdev_image = np.nanstd(clipped_cube,
                            axis=0)

    return mean_image, stdev_image


def pad_with_ref_pixels(image):
    """
    Pad the image with four rows and columns of (reference) pixels with value of zero.
    """
    padded_im = np.zeros((4096, 4096), dtype=np.uint32)
    padded_im[4:-4, 4:-4] = image

    return padded_im


def remove_ref_pixel_border(image):
    """
    Remove the outer four columns and rows of (reference) pixels to return the science image.
    """
    return image[4:-4, 4:-4]


def smooth_image(image, boxwidth):
    """
    Smooth the inputted image using Box2DKernel kernel at specified boxwidth.
    """
    smoothing_kernel = Box2DKernel(boxwidth)

    smoothed_image = convolve(image,
                              smoothing_kernel,
                              boundary="fill",
                              fill_value=np.nanmedian(image),
                              nan_treatment="interpolate")

    return smoothed_image


# I see that Flat and Dark have their own DataCube classes, should this
# be one of those? Potentially might need when identifying from Darks
# # Below is a slightly cleaned up version of ramp-fitting code from Bernie R.
# class Polynomial:
#     def __init__(self, nz: int, degree: int = 1):
#         self.nz = nz
#         self.degree = degree
#         self.z = np.arange(nz)

#         # Precompute basis matrix and its pseudo-inverse
#         self.B = np.vander(self.z, N=degree + 1, increasing=True)
#         self.pinvB = np.linalg.pinv(self.B)
#         self.B_x_pinvB = self.B @ self.pinvB

#     def fit(self, D):
#         """
#         Fits the polynomial to the data.
#         Returns the coefficients of the fit.
#         """
#         return (self.pinvB @ D.reshape(self.nz, -1)).reshape((-1, *D.shape[1:]))

#     def model(self, D):
#         """
#         Models the data based on the polynomial fit.
#         Returns the modeled data.
#         """
#         return (self.B_x_pinvB @ D.reshape(self.nz, -1)).reshape((-1, *D.shape[1:]))


# def get_slope(data):
#     """
#     Extracts the slope (linear term) of the data using polynomial fitting.
#     """
#     P = Polynomial(data.shape[0], degree=1)

#     # Extract the linear coefficient
#     slope = P.fit(data)[1]

#     return slope

class Polynomial():
    def __init__(self, nz:int, degree:int=1):
        # Setup
        self.nz = nz
        self.z = np.arange(nz)
        self.degree = degree
        # Make basis matrix
        self.B = np.empty((self.nz,degree+1))
        for col in np.arange(degree+1):
            self.B[:,col] = self.z**col
        # Make the fitting matrix. This does least squares fitting
        self.pinvB = np.linalg.pinv(self.B)
        # Make the modeling matrix. This is used to model the data
        # from the fit
        self.B_x_pinvB = np.matmul(self.B, self.pinvB)

    # Fitter
    def polyfit(self, D):
        # Print a warning if not float32
        # if D.dtype != np.float32:
        #     print('Warning: This operation is faster using np.float32 input')
        # Pick off dimensions
        ny = D.shape[1]
        nx = D.shape[2]
        # Fit
        R = np.matmul(self.pinvB, D.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(R)

    # Modeler
    def polyval(self, D):
        # Print a warning if not float32
        # if D.dtype != np.float32:
        #     print('Warning: This operation is faster using np.float32 input')
        ny = D.shape[1]
        nx = D.shape[2]
        # Model
        M = np.matmul(self.B_x_pinvB, D.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(M)

def get_slope(data):
    sh = data.shape
    nz = sh[0]
    ny = sh[1]
    nx = sh[2]
    P = Polynomial(nz, 1)
    S = np.empty((1,ny,nx))
    S[0] = P.polyfit(data)[1]
    return S[0, :,:]

def create_master_slope_image(filelist, sigma):
    """
    Perform reference pixel correction and create slope image for each file in filelist.
    Then, sigma-clip the stack to create a single, master slope image.
    """
    # Beginning by getting a list of slopes
    slopes = []

    for file in filelist:

        with rdm.open(file) as data:
            slope = get_slope(data.data.value)

        slopes.append(slope)

    # Creating a master slope image by sigma-clipping
    master_slope, _ = create_image_stats(slopes,
                                         sigma=sigma)

    return master_slope


def create_normalized_slope_image(filelist, sigma, boxwidth):
    """
    Create a normalized image by dividing the master-averaged slope image by
    the smoothed image. Used for DEAD, LOW_QE, OPEN/ADJ.
    """
    master_slope = create_master_slope_image(filelist,
                                             sigma=sigma)

    # Removing reference pixel border from image
    master_slope = remove_ref_pixel_border(master_slope)

    # Smoothing the image with the given boxwidth kernelsize
    smoothed_image = smooth_image(master_slope,
                                  boxwidth=boxwidth)

    # Returning the normalized image
    return master_slope / smoothed_image
