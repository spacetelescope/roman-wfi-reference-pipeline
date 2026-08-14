import logging
from math import atan2, hypot, sin

import numpy as np
import pysiaf
from astropy import units as u
from roman_datamodels.datamodels import PixelareaRefModel

from wfi_reference_pipeline.resources.wfi_meta_pixel_area import (
    WFIMetaPixelArea,
)

from ..reference_type import ReferenceType


class PixelArea(ReferenceType):
    """
    Class PixelArea() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.

    The pixel area map is generated from the Roman SIAF distortion
    polynomials. The Jacobian determinant of the distortion transform
    is evaluated across the detector and normalized by the nominal
    pixel area at the reference location.

    Examples
    --------

    from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import PixelArea
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta

    tmp = MakeDevMeta(ref_type="PIXELAREA")
    tmp.meta_pixelarea
    WFIMetaPixelArea(reference_type='PIXELAREA', pedigree='DUMMY', description='For RFP Development and DMS Build Support.', author='RFP', _use_after=<Time object: scale='utc' format='isot' value=2023-01-01T00:00:00.000>, telescope='ROMAN', origin='STSCI', instrument='WFI', instrument_detector='WFI01', pixelarea_steradians=np.float64(2.844036095230844e-13), pixelarea_arcsecsq=np.float64(0.0121))

    rfp_pam = PixelArea(meta_data=tmp.meta_pixelarea)


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

        # siaf_data = pysiaf.siaf.Siaf(instrument, filename=None, basepath=None, AperNames=None)
        # instrument = 'roman', filename = xml file name, basepath = path to where xml file is but without file name

        self.siaf = pysiaf.Siaf("Roman")

        detector = self.meta_data.instrument_detector

        if detector is None:
            raise ValueError(
                "instrument_detector must be supplied in metadata."
            )

        self.detector = detector.upper()

        self.x_coeffs = None
        self.y_coeffs = None
        self.nominal_pixel_area = None

        if ref_type_data is not None:

            if not isinstance(ref_type_data, np.ndarray):
                raise TypeError(
                    f"ref_type_data must be a numpy.ndarray, "
                    f"received {type(ref_type_data)}"
                )

            self.pixel_area = ref_type_data.astype(np.float32)

        else:
            self.pixel_area = self.make_pixel_area_image()

            logging.debug(
                f"Initialized PixelArea reference file: {outfile}"
            )

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

    def make_pixel_area_image(self,
                              include_border=False,
                              refpix_area=False,
                              ):
        """
        Generate a normalized pixel area map.

        Parameters
        ----------
        include_border : bool
            Include reference pixel border.

        refpix_area : bool
            If include_border=True, compute areas for reference pixels.
            Otherwise set them to zero.

        Returns
        -------
        numpy.ndarray
            Normalized pixel area map.
        """

        self.x_coeffs, self.y_coeffs = self.get_coeffs()

        nominal_area = self.get_nominal_area()

        det_size = 2048 if include_border else 2044

        pixels = np.mgrid[
            -det_size:det_size,
            -det_size:det_size,
        ]

        y = pixels[0]
        x = pixels[1]

        pixel_area = self.jacob(
            self.x_coeffs,
            self.y_coeffs,
            x,
            y,
            order=5,
        ).astype(np.float32)

        pixel_area /= nominal_area.value

        if include_border and not refpix_area:
            pixel_area[:4, :] = 0
            pixel_area[-4:, :] = 0
            pixel_area[:, :4] = 0
            pixel_area[:, -4:] = 0

        self.nominal_pixel_area = nominal_area

        self.meta_data.pixelarea_arcsecsq = (
            nominal_area.to(u.arcsec * u.arcsec).value
        )

        self.meta_data.pixelarea_steradians = (
            nominal_area.to(u.sr).value
        )

        return pixel_area

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the pixel area map reference.
        """
        pam_ref = PixelareaRefModel()
        pam_ref["meta"] = self.meta_data.export_asdf_meta()
        pam_ref["data"] = self.pixel_area.astype(np.float32)

        return pam_ref


    def get_coeffs(self):
        """
        Retrieve distortion coefficients from Roman SIAF.

        Returns
        -------
        tuple
            (x_coeffs, y_coeffs)
        """

        aperture_name = f"{self.detector}_FULL"

        coeffs = self.siaf[
            aperture_name
        ].get_polynomial_coefficients()

        return (
            coeffs["Sci2IdlX"],
            coeffs["Sci2IdlY"],
        )

    def get_nominal_area(self):
        """
        Compute nominal reference pixel area.

        Returns
        -------
        astropy.units.Quantity
        """

        x_scale = hypot(
            self.x_coeffs[1],
            self.y_coeffs[1],
        )

        y_scale = hypot(
            self.x_coeffs[2],
            self.y_coeffs[2],
        )

        bx = atan2(
            self.x_coeffs[1],
            self.y_coeffs[1],
        )

        pixel_area = (
            x_scale
            * y_scale
            * sin(bx)
            * u.arcsec
            * u.arcsec
        )

        return pixel_area

    @staticmethod
    def dpdx(a, x, y, order=5):
        """
        Partial derivative with respect to X.
        """

        partial_x = 0.0

        k = 1

        for i in range(1, order + 1):
            for j in range(i + 1):

                if i - j > 0:
                    partial_x += (
                        (i - j)
                        * a[k]
                        * x ** (i - j - 1)
                        * y ** j
                    )

                k += 1

        return partial_x

    @staticmethod
    def dpdy(a, x, y, order=5):
        """
        Partial derivative with respect to Y.
        """

        partial_y = 0.0

        k = 1

        for i in range(1, order + 1):
            for j in range(i + 1):

                if j > 0:
                    partial_y += (
                        j
                        * a[k]
                        * x ** (i - j)
                        * y ** (j - 1)
                    )

                k += 1

        return partial_y

    @classmethod
    def jacob(
        cls,
        a,
        b,
        x,
        y,
        order=5,
    ):
        """
        Compute Jacobian determinant.
        """

        jacobian = (
            cls.dpdx(a, x, y, order=order)
            * cls.dpdy(b, x, y, order=order)
            - cls.dpdx(b, x, y, order=order)
            * cls.dpdy(a, x, y, order=order)
        )

        return np.abs(jacobian)