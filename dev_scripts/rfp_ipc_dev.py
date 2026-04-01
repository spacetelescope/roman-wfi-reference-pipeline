import asdf
import numpy as np
import roman_datamodels.stnode as rds

from ..reference_type import ReferenceType


class InterPixelCapacitance(ReferenceType):
    """
    Class ipc() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.
    """

    def __init__(self, meta_data, user_ipc=None, file_list=None, outfile='roman_ipcfile.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables.
        """
        # Pass an empty list to satisfy the base class 'file_list' requirement
        if file_list is None:
            file_list = []
            
        # Passing explicitly as kwargs to dodge the base class reshuffle
        super().__init__(file_list=file_list, meta_data=meta_data, clobber=clobber)

        # Update metadata using the new 'self.meta_data' base class attribute
        if 'description' not in self.meta_data.keys():
            self.meta_data['description'] = 'Roman WFI interpixel capacitance reference file.'
        if 'reftype' not in self.meta_data.keys():
            self.meta_data['reftype'] = 'IPC'

        # Initialize attributes
        self.outfile = outfile
        self.ipc_kernel = user_ipc

    def make_ipc_kernel(self):
        """
        The method make_ipc_kernel() generates a WFI inter-pixel capacitance matrix or uses the values supplied
        when the class instance was initiated.
        """

        # IPC kernels derived by Analia Cillis (GSFC/CRESST II) from Flight FPS TVAC testing.
        ipc_kernels_default = {
            'WFI01': np.array([[0.0011245899, 0.0116328, 0.00116086], [0.0121407, 0.946092, 0.0121407], [0.0012092299, 0.012334201, 0.00118505]], dtype=np.float32),
            'WFI02': np.array([[0.0014147501, 0.0142855, 0.00143776], [0.0147226, 0.934254, 0.0150447], [0.0014262501, 0.0145156, 0.0014147501]], dtype=np.float32),
            'WFI03': np.array([[0.00135603, 0.013443399, 0.00135603], [0.0136187, 0.93804395, 0.013992799], [0.00137941, 0.014027899, 0.00137941]], dtype=np.float32),
            'WFI04': np.array([[0.00090399303, 0.010017199, 0.00090399303], [0.0113732, 0.951074, 0.0118741], [0.00100172, 0.0109823, 0.00101394]], dtype=np.float32),
            'WFI05': np.array([[0.0016382299, 0.015952999, 0.00167213], [0.0162693, 0.926483, 0.0166761], [0.00167213, 0.0162581, 0.0016608299]], dtype=np.float32),
            'WFI06': np.array([[0.00115506, 0.0124935, 0.00116685], [0.0129414, 0.941976, 0.0133539], [0.00132007, 0.0132361, 0.00132007]], dtype=np.float32),
            'WFI07': np.array([[0.0013906001, 0.0144075, 0.0014134], [0.013792, 0.93616897, 0.0142251], [0.0013564, 0.0144189, 0.00134501]], dtype=np.float32),
            'WFI08': np.array([[0.0014142699, 0.0134469, 0.0014484801], [0.0132644, 0.93911797, 0.0134926], [0.00142567, 0.0137207, 0.00139146]], dtype=np.float32),
            'WFI09': np.array([[0.00164033, 0.017787, 0.0017184401], [0.0168385, 0.921599, 0.0171398], [0.00169613, 0.018032499, 0.00162917]], dtype=np.float32),
            'WFI10': np.array([[0.0014237199, 0.013268599, 0.0014237199], [0.0133386, 0.939679, 0.013420301], [0.0014003799, 0.013467, 0.00141205]], dtype=np.float32),
            'WFI11': np.array([[0.00145249, 0.013507, 0.0014982399], [0.0139645, 0.937234, 0.0142733], [0.0014868, 0.013667099, 0.00142962]], dtype=np.float32),
            'WFI12': np.array([[0.0015740601, 0.0154165, 0.0016898001], [0.0152313, 0.929237, 0.015729], [0.00173609, 0.0160878, 0.00160878]], dtype=np.float32),
            'WFI13': np.array([[0.00145079, 0.0138182, 0.00140323], [0.0146268, 0.934905, 0.0149241], [0.0014388999, 0.014496, 0.0014983601]], dtype=np.float32),
            'WFI14': np.array([[0.00139219, 0.0129633, 0.00139219], [0.0131573, 0.940524, 0.0134654], [0.0013579499, 0.0130546, 0.0013693599]], dtype=np.float32),
            'WFI15': np.array([[0.00110193, 0.012729599, 0.00130854], [0.011375099, 0.94554603, 0.0117309], [0.0012626299, 0.0128444005, 0.00106749]], dtype=np.float32),
            'WFI16': np.array([[0.0015835101, 0.014423701, 0.00157204], [0.0142746, 0.934445, 0.0145041], [0.00153761, 0.0145959, 0.00156056]], dtype=np.float32),
            'WFI17': np.array([[0.0017083301, 0.016548, 0.0017197201], [0.015375, 0.926211, 0.0162519], [0.00169694, 0.0166847, 0.0017083301]], dtype=np.float32),
            'WFI18': np.array([[0.00173633, 0.0164837, 0.00172491], [0.0143703995, 0.92866206, 0.0149416], [0.0016106699, 0.0165751, 0.00163352]], dtype=np.float32),
        }

        if self.ipc_kernel is None:
            # Safely grab the detector using the new meta_data attribute
            detector = self.meta_data.get('instrument', {}).get('detector', 'WFI01')
            self.ipc_kernel = ipc_kernels_default.get(detector, ipc_kernels_default['WFI01'])
            print(f"Using default IPC kernel for {detector} from Cillis Flight FPS TVAC data.")
        else:
            if self.ipc_kernel.shape != (3, 3):
                print("The input dimensions of supplied IPC kernel are not 3x3.")

    # def populate_datamodel_tree(self):
    #     """
    #     Create data model from DMS and populate tree.
    #     """
    #     interpixelcapacitance_datamodel_tree = rds.IpcRef()
    #     interpixelcapacitance_datamodel_tree['meta'] = self.meta_data
    #     interpixelcapacitance_datamodel_tree['data'] = self.ipc_kernel

    #     return interpixelcapacitance_datamodel_tree
    
    def calculate_error(self):
        """
        Required abstract method implementation for ReferenceType.
        """
        pass

    def update_data_quality_array(self):
        """
        Required abstract method implementation for ReferenceType.
        """
        pass
    
    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """
        interpixelcapacitance_datamodel_tree = rds.IpcRef()
        
        # Convert the Trojan Horse meta object back into a raw dictionary
        # This prevents the ASDF yaml util from crashing on serialization
        clean_meta = dict(self.meta_data)
        
        # Ensure the 'reftype' key exists in the raw dictionary
        if 'reftype' not in clean_meta:
            clean_meta['reftype'] = getattr(self.meta_data, 'reference_type', 'IPC')
            
        interpixelcapacitance_datamodel_tree['meta'] = clean_meta
        interpixelcapacitance_datamodel_tree['data'] = self.ipc_kernel

        return interpixelcapacitance_datamodel_tree