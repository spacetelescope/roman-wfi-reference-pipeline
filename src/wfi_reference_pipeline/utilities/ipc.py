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

    def __init__(self, meta_data, user_ipc=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_ipcfile.asdf'
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        -------
        self.input_data: variable;
            The first positional variable in the IPC class instance assigned in base class ReferenceFile().
            For IPC() self.input_data must be a 3x3 numpy array.
        """

        # Access methods of base class ReferenceFile
        super(IPC, self).__init__(user_ipc, meta_data, clobber=clobber)

        # Update metadata with mask file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI inter-pixel capacitance reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'IPC'
        else:
            pass

        self.ipc_kernel = None

    def make_ipc_ref_file(self):
        """
        The method make_ipc_ref_file() generates a WFI inter-pixel capacitance reference file.'

        Outputs
        -------
        af: asdf file tree: {meta, data}
            meta:
            dq: mask - data quality array
                masked reference pixels flagged 2**31
                masked bad pixels flagged 2**1
        """

        ipc_kernel_default = np.array([[0.0021, 0.0166, 0.0022],
                               [0.0188, 0.9159, 0.0187],
                               [0.0021, 0.0162, 0.0020]], dtype=np.float32)

        if self.input_data is None:
            self.ipc_kernel = ipc_kernel_default
            print("Using default IPC kernel from Bellini et al. 2022.")
        else:
            if self.input_data.shape != (3, 3):
                print("The input dimensions of supplied IPC kernel is not 3x3.")
            else:
                self.ipc_kernel = self.input_data
                print("User supplied IPC kernel used to make reference file.")

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # Construct the dark object from the data model.
        ipc_file = rds.IpcRef()
        ipc_file['meta'] = self.meta
        ipc_file['data'] = self.ipc_kernel

        # af: asdf file tree: {meta, data, err, dq}
        af = asdf.AsdfFile()
        af.tree = {'roman': ipc_file}
        af.write_to(self.outfile)