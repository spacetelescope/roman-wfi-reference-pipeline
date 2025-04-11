import asdf
import numpy as np
import roman_datamodels.stnode as rds

from ..reference_type import ReferenceType


class InterPixelCapacitance(ReferenceType):
    """
    Class ipc() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method make_mask() creates the asdf mask file.
    """

    def __init__(self, meta_data, user_ipc=None, outfile='roman_ipcfile.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        -------
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = 'roman_ipcfile.asdf'
        self.input_data: variable;
            The first positional variable in the IPC class instance assigned in base class ReferenceType().
            For IPC() self.input_data must be a 3x3 numpy array.
        """

        # Access methods of base class ReferenceType
        super().__init__(user_ipc, meta_data, clobber=clobber)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI interpixel capacitance reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'IPC'

        # Initialize attributes
        self.outfile = outfile
        self.ipc_kernel = user_ipc

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

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        interpixelcapacitance_datamodel_tree = rds.IpcRef()
        interpixelcapacitance_datamodel_tree['meta'] = self.meta
        interpixelcapacitance_datamodel_tree['data'] = self.ipc_kernel

        return interpixelcapacitance_datamodel_tree

    def save_interpixelcapacitance(self, datamodel_tree=None):
        """
        The method save_interpixelcapacitance writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)
