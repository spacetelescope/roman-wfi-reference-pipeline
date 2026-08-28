from datetime import datetime, timedelta

from astropy import units as u

from wfi_reference_pipeline.constants import (
    DEFAULT_DESCRIPTION,
    REF_TYPE_DARK,
    REF_TYPE_DARKDECAYSIGNAL,
    REF_TYPE_DESCRIPTION,
    REF_TYPE_DETECTORSTATUS,
    REF_TYPE_EPSF,
    REF_TYPE_ETC,
    REF_TYPE_FGS_MASK,
    REF_TYPE_FLAT,
    REF_TYPE_GAIN,
    REF_TYPE_INTEGRALNONLINEARITY,
    REF_TYPE_INTERPIXELCAPACITANCE,
    REF_TYPE_INVERSELINEARITY,
    REF_TYPE_LINEARITY,
    REF_TYPE_MASK,
    REF_TYPE_PHOTOM,
    REF_TYPE_PIXELAREA,
    REF_TYPE_READNOISE,
    REF_TYPE_REFPIX,
    REF_TYPE_SATURATION,
    WFI_DETECTORS,
    WFI_MODE_WIM,
    WFI_PEDIGREE,
    WFI_REF_TYPES,
    WFI_TYPE_IMAGE,
)
from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark
from wfi_reference_pipeline.resources.wfi_meta_dark_decay_signal import (
    WFIMetaDarkDecaySignal,
)
from wfi_reference_pipeline.resources.wfi_meta_detector_status import (
    WFIMetaDetectorStatus,
)
from wfi_reference_pipeline.resources.wfi_meta_empirical_psf import (
    WFIMetaEPSF,
)
from wfi_reference_pipeline.resources.wfi_meta_exposure_time_calculator import (
    WFIMetaETC,
)
from wfi_reference_pipeline.resources.wfi_meta_fgs_mask import WFIMetaFGSMask
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat
from wfi_reference_pipeline.resources.wfi_meta_gain import WFIMetaGain
from wfi_reference_pipeline.resources.wfi_meta_integral_non_linearity import (
    WFIMetaIntegralNonLinearity,
)
from wfi_reference_pipeline.resources.wfi_meta_inter_pixel_capacitance import (
    WFIMetaInterPixelCapacitance,
)
from wfi_reference_pipeline.resources.wfi_meta_inverse_linearity import (
    WFIMetaInverseLinearity,
)
from wfi_reference_pipeline.resources.wfi_meta_linearity import WFIMetaLinearity
from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask
from wfi_reference_pipeline.resources.wfi_meta_photom import WFIMetaPhotom
from wfi_reference_pipeline.resources.wfi_meta_pixel_area import WFIMetaPixelArea
from wfi_reference_pipeline.resources.wfi_meta_readnoise import WFIMetaReadNoise
from wfi_reference_pipeline.resources.wfi_meta_referencepixel import (
    WFIMetaReferencePixel,
)
from wfi_reference_pipeline.resources.wfi_meta_saturation import WFIMetaSaturation


class MakeOpsMeta:
    """
    Class to generate any complete reference file MetaData object.

    Example Usage:
    from wfi_reference_pipeline.resources.make_ops_meta import MakeOpsMeta
    ops_meta_maker = MakeOpsMeta("DARK")
    dark_meta_data = ops_meta_maker.meta_dark
    """

    def _create_ops_meta_dark(self, meta_data):
        mode = WFI_MODE_WIM
        type = WFI_TYPE_IMAGE
        ref_optical_element = ["F158"]

        dark_meta_data = [mode, type, ref_optical_element]
        self.meta_dark = WFIMetaDark(*meta_data, *dark_meta_data)

    def _create_ops_meta_dark_decay_signal(self, meta_data):
        self.meta_dark_decay_signal = WFIMetaDarkDecaySignal(*meta_data)

    def _create_ops_meta_detector_status(self, meta_data):
        self.meta_detector_status = WFIMetaDetectorStatus(*meta_data)
    
    def _create_ops_meta_epsf(self, meta_data):
        ref_optical_element = "F062"

        oversample = 4
        spectral_type = ["A0V", "G2V", "M5V"]
        defocus = [0, 1, 2]

        pixel_x = [4.0, 2047.5, 4091.0,
                4.0, 2047.5, 4091.0,
                4.0, 2047.5, 4091.0]

        pixel_y = [4.0, 4.0, 4.0,
                2047.5, 2047.5, 2047.5,
                4091.0, 4091.0, 4091.0]

        # Required by schema but missing in file
        jitter_major = 1.0
        jitter_minor = 1.0
        jitter_position_angle = 1.0

        epsf_meta_data = [
            ref_optical_element,
            oversample,
            spectral_type,
            defocus,
            pixel_x,
            pixel_y,
            jitter_major,
            jitter_minor,
            jitter_position_angle,
        ]

        self.meta_epsf = WFIMetaEPSF(*meta_data, *epsf_meta_data)

    def _create_ops_meta_etc(self, meta_data):
        self.meta_etc = WFIMetaETC(*meta_data)

    def _create_ops_meta_fgs_mask(self, meta_data):
        self.meta_etc = WFIMetaFGSMask(*meta_data)

    def _create_ops_meta_flat(self, meta_data):
        ref_optical_element = "F158"

        flat_meta_data = [ref_optical_element]
        self.meta_flat = WFIMetaFlat(*meta_data, *flat_meta_data)

    def _create_ops_meta_gain(self, meta_data):
        self.meta_gain = WFIMetaGain(*meta_data)

    def _create_ops_meta_integral_non_linearity(self, meta_data):
        # There are 32 amplifiers to read 128 pixels at a time.
        # https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/coordinate-systems
        n_channels = 32
        n_pixels_per_channel = 128

        meta_integral_non_linearity = [n_channels, n_pixels_per_channel]
        self.meta_integral_non_linearity = WFIMetaIntegralNonLinearity(*meta_data,
                                                                       *meta_integral_non_linearity)

    def _create_ops_meta_inter_pixel_capacitance(self, meta_data):
        ref_optical_element = "F158"

        ipc_meta_data = [ref_optical_element]
        self.meta_ipc = WFIMetaInterPixelCapacitance(*meta_data, *ipc_meta_data)

    def _create_ops_meta_inverse_linearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        inverselinearity_meta_data = [input_units, output_units]
        self.meta_inverselinearity = WFIMetaInverseLinearity(*meta_data,
                                                             *inverselinearity_meta_data)

    def _create_ops_meta_linearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        linearity_meta_data = [input_units, output_units]
        self.meta_linearity = WFIMetaLinearity(*meta_data, *linearity_meta_data)

    def _create_ops_meta_mask(self, meta_data):
        self.meta_mask = WFIMetaMask(*meta_data)

    def _create_ops_meta_pixelarea(self, meta_data):
        p_optical_element = "F158"  # Default optical element

        # pixel scale (Roman WFI is ~0.11 arcsec/pixel)
        pixel_scale = 0.11 * u.arcsec
        # Pixel area in arcsecsq
        pixelarea_arcsecsq = (pixel_scale ** 2).to(u.arcsec**2).value
        # Convert to steradians
        pixelarea_steradians = (pixel_scale ** 2).to(u.sr).value

        self.meta_pixelarea = WFIMetaPixelArea(
            *meta_data,
            pixelarea_steradians=pixelarea_steradians,
            pixelarea_arcsecsq=pixelarea_arcsecsq,
            ref_optical_element=p_optical_element,
            )

    def _create_ops_meta_photom(self, meta_data):
        self.meta_photom = WFIMetaPhotom(*meta_data)

    def _create_ops_meta_readnoise(self, meta_data):
        mode = WFI_MODE_WIM
        type = WFI_TYPE_IMAGE

        readnoise_meta_data = [mode, type]
        self.meta_readnoise = WFIMetaReadNoise(*meta_data, *readnoise_meta_data)

    def _create_ops_meta_referencepixel(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        referencepixel_meta_data = [input_units, output_units]
        self.meta_referencepixel = WFIMetaReferencePixel(*meta_data,
                                                         *referencepixel_meta_data)

    def _create_ops_meta_saturation(self, meta_data):
        self.meta_saturation = WFIMetaSaturation(*meta_data)

    def __init__(self, ref_type, routine_delivery_type=True):
        """
        Generates a reference type specific MetaData object relevant to the ref_type
        parameter.

        Parameters
        -------
        ref_type: str;
            String defining the reference file type which will determine the reference
            meta object created.
        routine_delivery_type: boolean;
            A True or False setting for routine high cadence delivery by the RFP, such 
            as weekly darks or monthly flats, vs low cadence once a year deliveries such
            as yearly linearity reference file.
            #TODO Work out with service accounts and automated pipeline run starts. Consider
            using either routine = True, or have a cadence variable like weekly, monthly or other.

        description notes:

        This first bit below is for standard routine deliveries and is the reason for delivery
        typically posted in the CRDS context update for the description for the delivery.
            "Delivering (18) new WFI dark reference files for imaging and spectral "
            "modes, WIM and WSM. "
            "This delivery is a weekly routine dark reference file delivery for data "
            "from 2026-07-08 through 2026-07-15. "

        The next part of the description is for the specific files. Below is a detailed exampled
        of what should be considered the gold standard for RFP and RTB delvieries to CRDS.
            "Dark calibration reference file containing the dark slope, dark slope "
            "error, and DQ arrays derived from the TVAC1 and TVAC2 Thermal Vacuum "
            "Tests of the Roman Wide Field Instrument (WFI). The calibration combines "
            "the Total Noise (OTP00639) and Dark (OTP00644) datasets consisting of "
            "100 55-frame dark exposures with 3.04 s frame times for a 170.3 s total "
            "exposure from TVAC1, plus 100 55-frame dark exposures with 3.16 s frame "
            "times for a 177.1 s total exposure from TVAC2, and 4 350-frame long dark "
            "exposures with 3.16 s frame times for an approximately 1110 s total "
            "exposure from the TVAC2 Dark dataset, all acquired during the Nominal "
            "Operations environmental plateau using the flight detectors and flight "
            "focal plane electronics."
        """

        # TODO check how to assign useafter to ref files taken throughout the week
        date_now = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        date_start = date_now - timedelta(days=7)

        ref_type_name = REF_TYPE_DESCRIPTION[ref_type]


        if routine_delivery_type:
            reason_for_delivery_string = (
                f"Delivering (18) new WFI {ref_type_name} reference files for imaging "
                f"and spectral modes, WIM and WSM. "
                f"This is a routine {ref_type_name} reference file "
                f"delivery for data from {date_start:%Y-%m-%d} through "
                f"{date_now:%Y-%m-%d}. "
            )
        else:
            reason_for_delivery_string = (
                f"Delivering (18) new WFI {ref_type_name} reference files for imaging "
                f"and spectral modes, WIM and WSM. "
                f"This is a {ref_type_name} reference file "
                f"delivery for data from {date_start:%Y-%m-%d} through "
                f"{date_now:%Y-%m-%d}. "
            )

        pedigree = "INFLIGHT"
        description = reason_for_delivery_string + DEFAULT_DESCRIPTION
        author = "RFP Version"
        try:
            use_after = date_start.strftime("%Y-%m-%dT%H:%M:%S.000")
        except (AttributeError, ValueError):
            use_after = "2026-12-01T00:00:00.000"
        telescope = "ROMAN"
        origin = "STSCI/SOC"
        instrument = "WFI"
        detector = "WFI01"  # Default - needs to be updated and checked for each instance

        """
        TODO for later - check if default description is still in meta data and wasn't changed. Raise warning or error.
        if DEFAULT_DESCRIPTION in description:
            warnings.warn("Using the default placeholder description.")
        
        """


        if ref_type not in WFI_REF_TYPES:
            raise ValueError(f"ref_type must be one of: {WFI_REF_TYPES}")
        if pedigree not in WFI_PEDIGREE:
            raise ValueError(f"pedigree must be one of: {WFI_PEDIGREE}")
        if detector not in WFI_DETECTORS:
            raise ValueError(f"detector must be one of: {WFI_DETECTORS}")

        meta_data_params = [ref_type, pedigree, description, author,
                            use_after, telescope, origin, instrument, detector]

        if ref_type == REF_TYPE_DARK:
            self._create_ops_meta_dark(meta_data_params)

        if ref_type == REF_TYPE_DARKDECAYSIGNAL:
            self._create_ops_meta_dark_decay_signal(meta_data_params)

        if ref_type == REF_TYPE_DETECTORSTATUS:
            self._create_ops_meta_detector_status(meta_data_params)
  
        if ref_type == REF_TYPE_EPSF:
            self._create_ops_meta_epsf(meta_data_params)

        if ref_type == REF_TYPE_ETC:
            self._create_ops_meta_etc(meta_data_params)

        if ref_type == REF_TYPE_FGS_MASK:
            self._create_ops_meta_fgs_mask(meta_data_params)

        if ref_type == REF_TYPE_FLAT:
            self._create_ops_meta_flat(meta_data_params)

        if ref_type == REF_TYPE_GAIN:
            self._create_ops_meta_gain(meta_data_params)

        if ref_type == REF_TYPE_INTEGRALNONLINEARITY:
            self._create_ops_meta_integral_non_linearity(meta_data_params)

        if ref_type == REF_TYPE_INVERSELINEARITY:
            self._create_ops_meta_inverse_linearity(meta_data_params)

        if ref_type == REF_TYPE_INTERPIXELCAPACITANCE:
            self._create_ops_meta_inter_pixel_capacitance(meta_data_params)

        if ref_type == REF_TYPE_LINEARITY:
            self._create_ops_meta_linearity(meta_data_params)

        if ref_type == REF_TYPE_MASK:
            self._create_ops_meta_mask(meta_data_params)

        if ref_type == REF_TYPE_PIXELAREA:
            self._create_ops_meta_pixelarea(meta_data_params)

        if ref_type == REF_TYPE_PHOTOM:
            self._create_ops_meta_photom(meta_data_params)

        if ref_type == REF_TYPE_READNOISE:
            self._create_ops_meta_readnoise(meta_data_params)

        if ref_type == REF_TYPE_REFPIX:
            self._create_ops_meta_referencepixel(meta_data_params)

        if ref_type == REF_TYPE_SATURATION:
            self._create_ops_meta_saturation(meta_data_params)