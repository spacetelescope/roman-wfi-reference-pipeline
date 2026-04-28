import logging
import numpy as np
from roman_datamodels.datamodels import IpcRefModel

from wfi_reference_pipeline.resources.wfi_meta_inter_pixel_capacitance import WFIMetaInterPixelCapacitance
from ..reference_type import ReferenceType

log = logging.getLogger(__name__)


class InterPixelCapacitance(ReferenceType):
    """
    Creates a Roman WFI IPC reference file containing a 3x3 kernel provided in the dictionary
    after the IPC class.

    example code:
    from wfi_reference_pipeline.reference_types.inter_pixel_capacitance.inter_pixel_capacitance import InterPixelCapacitance
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
    tmp = MakeDevMeta(ref_type='IPC')
    rfp_ipc = InterPixelCapacitance(meta_data=tmp.meta_ipc)
    rfp_ipc.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        ref_type_data=None,
        outfile="roman_ipc.asdf",
        clobber=False,
    ):
        super().__init__(
            meta_data,
            ref_type_data=ref_type_data,
            outfile=outfile,
            clobber=clobber,
        )

        # Metadata validation
        if not isinstance(meta_data, WFIMetaInterPixelCapacitance):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaInterPixelCapacitance"
            )

        # Default description
        if not getattr(self.meta_data, "description", ""):
            self.meta_data.description = (
                "Roman WFI interpixel capacitance (IPC) reference file."
            )

        if ref_type_data is not None:
            self.ipc_kernel = ref_type_data
            if not isinstance(self.ipc_kernel, np.ndarray):
                raise TypeError(
                    f"ref_type_data must be a numpy.ndarray, got {type(self.ipc_kernel)}"
                )

            if self.ipc_kernel.shape != (3, 3):
                raise ValueError(
                    f"IPC kernel must be 3x3, got shape {self.ipc_kernel.shape}"
                )

            msg = "Using user-supplied IPC kernel."
            log.info(msg)
            print(msg)

            self.ipc_kernel.astype(np.float32)
        
        else:
            self.ipc_kernel = self._get_ipc_kernel()
        

    def _get_ipc_kernel(self):
        """
        Get detector specific from IPC kernel detector lookup dictionary.
        """

        detector = self.meta_data.instrument_detector
        try:
            ipc_kernel = IPC_KERNELS[detector]
        except KeyError:
            raise KeyError(
                f"Detector '{detector}' not found in IPC_KERNELS."
            )

        msg = (
            f"No user IPC kernel provided. "
            f"Looking up IPC kernel for detector {detector} from internal dictionary."
        )
        log.info(msg)
        print(msg)

        return ipc_kernel

    def populate_datamodel_tree(self):
        """
        Create the datamodel tree to be written to ASDF.
        """
        ipc_datamodel_tree = IpcRefModel()
        ipc_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        ipc_datamodel_tree["data"] = self.ipc_kernel

        return ipc_datamodel_tree

    def calculate_error(self):
        """Not utilized."""
        pass

    def update_data_quality_array(self):
        """Not utilized."""
        pass

# =====================================================================
# IPC kernels derived by A. Cillis (GSFC/CRESST II)
# From Flight FPS TVAC testing, coded by H. Khandrika, and
# updated by R. Cosentino April 2026. 
# =====================================================================
IPC_KERNELS = {
    "WFI01": np.array([
        [0.0011245899, 0.0116328,  0.00116086],
        [0.0121407,    0.946092,   0.0121407],
        [0.0012092299, 0.012334201, 0.00118505],
    ], dtype=np.float32),

    "WFI02": np.array([
        [0.0014147501, 0.0142855,  0.00143776],
        [0.0147226,    0.934254,   0.0150447],
        [0.0014262501, 0.0145156,  0.0014147501],
    ], dtype=np.float32),

    "WFI03": np.array([
        [0.00135603,  0.013443399, 0.00135603],
        [0.0136187,   0.93804395,  0.013992799],
        [0.00137941,  0.014027899, 0.00137941],
    ], dtype=np.float32),

    "WFI04": np.array([
        [0.00090399303, 0.010017199, 0.00090399303],
        [0.0113732,     0.951074,    0.0118741],
        [0.00100172,    0.0109823,   0.00101394],
    ], dtype=np.float32),

    "WFI05": np.array([
        [0.0016382299, 0.015952999, 0.00167213],
        [0.0162693,    0.926483,    0.0166761],
        [0.00167213,   0.0162581,   0.0016608299],
    ], dtype=np.float32),

    "WFI06": np.array([
        [0.00115506, 0.0124935,  0.00116685],
        [0.0129414,  0.941976,   0.0133539],
        [0.00132007, 0.0132361,  0.00132007],
    ], dtype=np.float32),

    "WFI07": np.array([
        [0.0013906001, 0.0144075, 0.0014134],
        [0.013792,     0.93616897, 0.0142251],
        [0.0013564,    0.0144189, 0.00134501],
    ], dtype=np.float32),

    "WFI08": np.array([
        [0.0014142699, 0.0134469, 0.0014484801],
        [0.0132644,    0.93911797, 0.0134926],
        [0.00142567,   0.0137207, 0.00139146],
    ], dtype=np.float32),

    "WFI09": np.array([
        [0.00164033,   0.017787,   0.0017184401],
        [0.0168385,    0.921599,   0.0171398],
        [0.00169613,   0.018032499, 0.00162917],
    ], dtype=np.float32),

    "WFI10": np.array([
        [0.0014237199, 0.013268599, 0.0014237199],
        [0.0133386,    0.939679,    0.013420301],
        [0.0014003799, 0.013467,    0.00141205],
    ], dtype=np.float32),

    "WFI11": np.array([
        [0.00145249,  0.013507,   0.0014982399],
        [0.0139645,   0.937234,   0.0142733],
        [0.0014868,   0.013667099, 0.00142962],
    ], dtype=np.float32),

    "WFI12": np.array([
        [0.0015740601, 0.0154165,  0.0016898001],
        [0.0152313,    0.929237,   0.015729],
        [0.00173609,   0.0160878,  0.00160878],
    ], dtype=np.float32),

    "WFI13": np.array([
        [0.00145079,  0.0138182,  0.00140323],
        [0.0146268,   0.934905,   0.0149241],
        [0.0014388999, 0.014496,  0.0014983601],
    ], dtype=np.float32),

    "WFI14": np.array([
        [0.00139219,  0.0129633,  0.00139219],
        [0.0131573,   0.940524,   0.0134654],
        [0.0013579499, 0.0130546, 0.0013693599],
    ], dtype=np.float32),

    "WFI15": np.array([
        [0.00110193,   0.012729599, 0.00130854],
        [0.011375099,  0.94554603,  0.0117309],
        [0.0012626299, 0.0128444005, 0.00106749],
    ], dtype=np.float32),

    "WFI16": np.array([
        [0.0015835101, 0.014423701, 0.00157204],
        [0.0142746,    0.934445,    0.0145041],
        [0.00153761,   0.0145959,   0.00156056],
    ], dtype=np.float32),

    "WFI17": np.array([
        [0.0017083301, 0.016548,  0.0017197201],
        [0.015375,     0.926211,  0.0162519],
        [0.00169694,   0.0166847, 0.0017083301],
    ], dtype=np.float32),

    "WFI18": np.array([
        [0.00173633,   0.0164837, 0.00172491],
        [0.0143703995, 0.92866206, 0.0149416],
        [0.0016106699, 0.0165751, 0.00163352],
    ], dtype=np.float32),
}
