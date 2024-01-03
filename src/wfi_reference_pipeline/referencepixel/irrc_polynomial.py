'''


@author: B. Rauscher
'''
import numpy as np

class Polynomial():
    """
    Highly efficient fitting and modeling of up-the-ramp sampled IR array data
    """
    
    def __init__(self, nz:int, degree:int=1):
        """
        Parameters: nz:int
                      Number of up-the-ramp samples
                    degree:int
                      Desired fit degree. Use
                      degree=1 for straight lines.
        """
        # Setup
        self.nz = nz
        self.z = np.arange(nz)
        self.degree = degree
        
        # Make basis matrix
        self.bb = np.empty((self.nz,degree+1))
        for col in np.arange(degree+1):
            self.bb[:,col] = self.z**col
            
        # Make the fitting matrix. This does least squares fitting
        self.pinv_b = np.linalg.pinv(self.bb)
        
        # Make the modeling matrix. This is used to model the data
        # from the fit
        self.b_x_pinvb = np.matmul(self.bb, self.pinv_b)
        
    # Fitter
    def polyfit(self, dd):
        """
        Fit polynomial to up-the-ramp sampled data
        
        Parameters: dd, data cube
                      An up-the-ramp sampled data cube.
        Returns:    rr, data cube
                      Least squares fit of a polynomial to the data
        """
        # Print a warning if not float32
        if dd.dtype != np.float32:
            print('Warning: This operation is faster using np.float32 input')
        # Pick off dimensions
        ny = dd.shape[1]
        nx = dd.shape[2]
        # Fit
        rr = np.matmul(self.pinv_b, dd.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(rr)
    
    # Modeler
    def polyval(self, dd):
        """
        Model up-the-ramp sampled data using the fit
        
        Parameters: dd, data cube
                      An up-the-ramp sampled data cube.
        Returns:    mm, data cube
                      A model of the data built from the least squares fit
        """
        # Print a warning if not float32
        if dd.dtype != np.float32:
            print('Warning: This operation is faster using np.float32 input')
        ny = dd.shape[1]
        nx = dd.shape[2]
        # Model
        mm = np.matmul(self.b_x_pinvb, dd.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(mm)