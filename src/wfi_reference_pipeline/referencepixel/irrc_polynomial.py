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
        """
        Fit polynomial to up-the-ramp sampled data
        
        Parameters: D, data cube
                      An up-the-ramp sampled data cube.
        Returns:    R, data cube
                      Least squares fit of a polynomial to the data
        """
        # Print a warning if not float32
        if D.dtype != np.float32:
            print('Warning: This operation is faster using np.float32 input')
        # Pick off dimensions
        ny = D.shape[1]
        nx = D.shape[2]
        # Fit
        R = np.matmul(self.pinvB, D.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(R)
    
    # Modeler
    def polyval(self, D):
        """
        Model up-the-ramp sampled data using the fit
        
        Parameters: D, data cube
                      An up-the-ramp sampled data cube.
        Returns:    M, data cube
                      A model of the data built from the least squares fit
        """
        # Print a warning if not float32
        if D.dtype != np.float32:
            print('Warning: This operation is faster using np.float32 input')
        ny = D.shape[1]
        nx = D.shape[2]
        # Model
        M = np.matmul(self.B_x_pinvB, D.reshape(self.nz,-1)).reshape((-1,ny,nx))
        return(M)