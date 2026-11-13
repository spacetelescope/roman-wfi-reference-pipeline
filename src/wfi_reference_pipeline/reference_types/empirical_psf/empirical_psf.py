import numpy as np
from roman_datamodels.datamodels import EpsfRefModel

from wfi_reference_pipeline.resources.wfi_meta_empirical_psf import (
    WFIMetaEPSF,
)

from ..reference_type import ReferenceType


class EmpiricalPSF(ReferenceType):
    """Class EmpiricalPSF() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.

    Currently, there is a psf library generator that calls stpsf to simulate
    psf reference files. This class mimics a workflow where the psf library is generated
    from outside the class and is input as an object to be inserted into the data model.

    A development script can produce the files - rfp_epsf_creation.py
    """

    def __init__(
        self,
        meta_data,
        psf,
        extended_psf=None,
        outfile="roman_epsf_file.asdf",
        clobber=False,
    ):

        super().__init__(
            meta_data=meta_data,
            ref_type_data=psf,
            outfile=outfile,
            clobber=clobber,
        )

        if not isinstance(meta_data, WFIMetaEPSF):
            raise TypeError(
                f"Meta Data has reftype "
                f"{type(meta_data)}, "
                f"expecting WFIMetaEPSF"
            )
        
        if not isinstance(psf, np.ndarray):
            raise TypeError(
                f"PSF data has type {type(psf)}, expecting numpy.ndarray"
                )

        self.meta = meta_data
        self.outfile = outfile
        self.psf = psf
        self.extended_psf = extended_psf

        self._update_meta_data()


    def _update_meta_data(self):
        """
        Update meta data for pixel lists if different from what is the standard 
        in the dev meta maker.
        """

        self.meta.pixel_x = list(
            self.meta.pixel_x
        )

        self.meta.pixel_y = list(
            self.meta.pixel_y
        )

        self.meta.spectral_type = list(
            self.meta.spectral_type
        )

        self.meta.defocus = list(
            self.meta.defocus
        )


    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the EPSF reference file.
        """

        epsf_datamodel_tree = EpsfRefModel()
        epsf_datamodel_tree["meta"] = self.meta.export_asdf_meta()
        epsf_datamodel_tree["psf"] = self.psf

        if self.extended_psf is not None:
            epsf_datamodel_tree[
                "extended_psf"
            ] = self.extended_psf

        return epsf_datamodel_tree
    
    def calculate_error(self):
        """
        Abstract method not applicable.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass