import logging
 
import numpy as np
from roman_datamodels.datamodels import BamRefModel
 
from wfi_reference_pipeline.resources.wfi_meta_boresight_alignment_matrix import (
    WFIMetaBoresightAlignmentMatrix,
)
 
from ..reference_type import ReferenceType
 
log = logging.getLogger(__name__)
 
 
class BoresightAlignmentMatrix(ReferenceType):
    """
    Creates a Roman WFI Boresight Alignment Matrix (BAM) reference file that stores
    the quaternion for the alignment matrix between the Roman observatory boresight
    and the fine guidance system (FGS).
 
    No array maps are created — this is a single observatory level quaternion that
    is agnostic of the detector, so instrument.detector is written as NOT_CONFIGURED.
    The meta data also carries the boresight V2, V3 (arcsec) and V3 Y angle (deg).
 
    The quaternion may be provided through ref_type_data as a list, tuple, or numpy
    array of 4 finite numbers. If ref_type_data is None, the default BAM_QUATERNION
    below (on-orbit values from Roman commissioning) is used.
 
    Example code to generate file:
    from wfi_reference_pipeline.reference_types.boresight_alignment_matrix.boresight_alignment_matrix import (
        BoresightAlignmentMatrix,
    )
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
 
    tmp = MakeDevMeta(ref_type='BAM')
    tmp.meta_bam.pedigree = 'INFLIGHT 2026-09-18 2026-09-20'
    tmp.meta_bam.use_after = '2026-09-01T00:00:00.000'
    tmp.meta_bam.author = 'T. Sohn, T. Desjardins'
    tmp.meta_bam.origin = 'STSCI/SOC'
    tmp.meta_bam.v2 = 0.0
    tmp.meta_bam.v3 = 0.0
    tmp.meta_bam.v3_yangle = 0.0
    tmp.meta_bam.description = 'Quaternion values for the alignment matrix between the Roman
    observatory boresight and fine guidance system (FGS).'
    rfp_bam = BoresightAlignmentMatrix(meta_data=tmp.meta_bam, ref_type_data=BAM_QUATERNION)
    rfp_bam.generate_outfile()
    """
 
    def __init__(self,
                 meta_data,
                 ref_type_data=None,
                 outfile="roman_bam.asdf",
                 clobber=False,
                 norm_tolerance=1.0e-6,
    ):
        """
        Parameters
        ----------
        meta_data: WFIMetaBoresightAlignmentMatrix
            Meta information object for the BAM reference file.
        ref_type_data: list | tuple | numpy.ndarray, default = None
            Quaternion with 4 elements. If None, BAM_QUATERNION is used.
        outfile: str
            Output ASDF file name.
        clobber: bool
            Whether to overwrite existing ASDF file.
        norm_tolerance: float
            Allowed deviation of the quaternion norm from 1.
 
        Not included
        ----------
        file_list: list[str] | None
        bit_mask: 2D integer numpy array, default = None
        """
        super().__init__(meta_data,
                         ref_type_data=ref_type_data,
                         outfile=outfile,
                         clobber=clobber)
 
        # Metadata validation
        if not isinstance(meta_data, WFIMetaBoresightAlignmentMatrix):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaBoresightAlignmentMatrix"
            )
 
        # Default description
        if not getattr(self.meta_data, "description", ""):
            self.meta_data.description = (
                "Roman WFI Boresight Alignment Matrix (BAM) reference file."
            )
 
        if ref_type_data is None:
            log.debug("No quaternion provided, using default BAM_QUATERNION.")
            ref_type_data = BAM_QUATERNION
 
        self.bam_quaternion = self._validate_bam_quaternion(ref_type_data,
                                                            norm_tolerance=norm_tolerance)
        self.outfile = outfile
 
    @staticmethod
    def _validate_bam_quaternion(quaternion, norm_tolerance=1.0e-6):
        """
        Validate a BAM quaternion and return it as a float64 numpy array.
 
        Parameters
        ----------
        quaternion: list | tuple | numpy.ndarray
            Quaternion with 4 finite numeric elements.
        norm_tolerance: float
            Allowed deviation of the quaternion norm from 1.
 
        Raises
        ------
        TypeError
            If the input is not a list, tuple, or numpy array of numbers.
        ValueError
            If the input does not have 4 finite elements or is not unit norm.
        """
        if not isinstance(quaternion, (list, tuple, np.ndarray)):
            raise TypeError(
                f"BAM quaternion must be a list, tuple, or numpy.ndarray, got {type(quaternion)}."
            )
 
        try:
            q = np.asarray(quaternion, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise TypeError(f"BAM quaternion must contain only numbers: {err}") from err
 
        if q.ndim != 1 or q.size != 4:
            raise ValueError(f"BAM quaternion must have exactly 4 elements, got shape {q.shape}.")
 
        if not np.all(np.isfinite(q)):
            raise ValueError("BAM quaternion contains non-finite values.")
 
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > norm_tolerance:
            raise ValueError(
                f"BAM quaternion must be unit norm within {norm_tolerance}, got norm {norm}."
            )
 
        return q
 
    def populate_datamodel_tree(self):
        """
        Create the datamodel tree to be written to ASDF.
        """
        bam_datamodel_tree = BamRefModel()
        bam_datamodel_tree["meta"] = self.meta_data.export_asdf_meta()
        bam_datamodel_tree["bam_quaternion"] = [float(q) for q in self.bam_quaternion]
 
        return bam_datamodel_tree
 
    def calculate_error(self):
        """
        Abstract method not utilized.
        """
        pass
 
    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass
 