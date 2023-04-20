import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np


class IPC(ReferenceFile):
    """
    Class ipc() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.
    """

    def __init__(self, meta_data, user_ipc=None, outfile='roman_ipcfile.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        -------
        outfile: string; default = 'roman_ipcfile.asdf'
        self.input_data: variable;
            The first positional variable in the IPC class instance assigned in base class ReferenceFile().
            For IPC() self.input_data must be a 3x3 numpy array.
        """

        # Access methods of base class ReferenceFile
        super().__init__(user_ipc, meta_data, clobber=clobber)

        # Update metadata with mask file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI inter-pixel capacitance reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'IPC'

        self.outfile = outfile
        self.ipc_kernel = user_ipc
        self.ipc_obj = None

    def make_ipc_kernel(self):
        """
        The method make_ipc_kernel() generates a WFI inter-pixel capacitance matrix or uses the values supplied
        when the class instance was initiated.'
        """

        # IPC kernel from Bellini et al. 2022 WFIsim: The Roman Telescope Branch
        # Wide-Field-Instrument Simulator - Nancy Grace Roman Space Telescope
        # Technical Report Roman-STScI-000433
        ipc_kernel_default = np.array([[0.0021, 0.0166, 0.0022],
                                       [0.0188, 0.9159, 0.0187],
                                       [0.0021, 0.0162, 0.0020]], dtype=np.float32)

        if self.ipc_kernel is None:
            self.ipc_kernel = ipc_kernel_default
            print("Using default IPC kernel from Bellini et al. 2022.")
        else:
            if self.ipc_kernel.shape != (3, 3):
                print("The input dimensions of supplied IPC kernel are not 3x3.")

    def make_ipc_obj(self):
        """
        The method make_ipc_obj creates an object from the DMS data model.
        """

        # Construct the dark object from the data model.
        self.ipc_obj = rds.IpcRef()
        self.ipc_obj['meta'] = self.meta
        self.ipc_obj['data'] = self.ipc_kernel

    def save_ipc(self):
        """
        The method save_ipc writes the reference file object to the specified asdf outfile.
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # af: asdf file tree: {meta, data, err, dq}
        af = asdf.AsdfFile()
        af.tree = {'roman': self.ipc_obj}
        af.write_to(self.outfile)