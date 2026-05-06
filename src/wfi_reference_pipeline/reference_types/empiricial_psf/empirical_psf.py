import asdf
import numpy as np
import roman_datamodels.stnode as rds

from ..reference_type import ReferenceType
from wfi_reference_pipeline.resources.wfi_meta_empirical_psf import WFIMetaEPSF

class EmpiricalPSF(ReferenceType):
    """
    Class EmpiricalPSF() inherits the ReferenceType() base class methods.
    This class builds Roman WFI EPSF reference files.

    The empirical point spread function reference file shall be made
    to do the following....
    """

    def __init__(
        self,
        meta_data,
        psf=None,
        extended_psf=None,
        outfile='roman_epsf_file.asdf',
        clobber=False,
    ):
        """
        Parameters
        -------
        meta_data : dict
            Metadata dictionary formatted per EPSF schema.
        psf : np.ndarray
            5D array (defocus, spectral_type, grid, y, x)
        extended_psf : np.ndarray, optional
            2D extended PSF
        """

        # Initialize base class
        super().__init__(psf, meta_data, clobber=clobber)

        # Ensure required metadata
        if 'description' not in self.meta:
            self.meta['description'] = 'Roman WFI Empirical PSF reference file.'

        if not isinstance(meta_data, WFIMetaEPSF):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaEPSF"
            )

        # Assign attributes
        self.outfile = outfile
        self.psf = psf
        self.extended_psf = extended_psf

    def make_psf_library(self):
        """
        Method to populate arrays
        """
    
        self._update_meta_data()

    def _update_meta_data(self):
        """
        Update the meta data with how the psf library was constructed.
        """

    # ------------------------------------------------------------------
    # Populate ASDF datamodel
    # ------------------------------------------------------------------
    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        epsf_datamodel_tree = rds.EpsfRef()

        # Required
        epsf_datamodel_tree['meta'] = self.meta
        epsf_datamodel_tree['psf'] = self.psf

        # Optional arrays
        if self.extended_psf is not None:
            epsf_datamodel_tree['extended_psf'] = self.extended_psf

        if self.psf_noipc is not None:
            epsf_datamodel_tree['psf_noipc'] = self.psf_noipc

        if self.extended_psf_noipc is not None:
            epsf_datamodel_tree['extended_psf_noipc'] = self.extended_psf_noipc

        return epsf_datamodel_tree