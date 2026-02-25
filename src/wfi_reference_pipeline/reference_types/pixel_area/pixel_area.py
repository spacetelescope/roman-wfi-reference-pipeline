import numpy as np
import roman_datamodels.stnode as rds

from wfi_reference_pipeline.resources.wfi_meta_pixel_area import WFIMetaPixelArea

from ..reference_type import ReferenceType


class PixelArea(ReferenceType):
    """
    Class PixelArea() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_pixelarea.asdf",
            clobber=False,
    ):

        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaPixelArea):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaPIXELAREA"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI pixel area reference file."

        if ref_type_data is None:
            ref_type_data = make_pam_array
        self.pixel_area = ref_type_data    

        self.outfile = outfile    

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

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the pixel area map reference.
        """
        pam_ref = rds.PixelareaRef()
        pam_ref["meta"] = self.meta_data.export_asdf_meta()
        pam_ref["data"] = self.pixel_area


def make_pam_array():

    arr = np.random.uniform(0.98, 1.02, size=(4096, 4096))

    return arr