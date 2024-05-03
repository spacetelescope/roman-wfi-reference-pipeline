from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark
from wfi_reference_pipeline.resources.wfi_meta_flat import WFIMetaFlat
from wfi_reference_pipeline.resources.wfi_meta_gain import WFIMetaGain
from wfi_reference_pipeline.resources.wfi_meta_inverselinearity import WFIMetaInverseLinearity
from wfi_reference_pipeline.resources.wfi_meta_interpixelcapacitance import WFIMetaIPC
from wfi_reference_pipeline.resources.wfi_meta_linearity import WFIMetaLinearity
from wfi_reference_pipeline.resources.wfi_meta_mask import WFIMetaMask
from wfi_reference_pipeline.resources.wfi_meta_readnoise import WFIMetaReadNoise
from wfi_reference_pipeline.resources.wfi_meta_referencepixel import WFIMetaReferencePixel
from wfi_reference_pipeline.resources.wfi_meta_saturation import WFIMetaSaturation
from wfi_reference_pipeline.constants import WFI_DETECTORS, WFI_MODE_WIM, WFI_PEDIGREE
from wfi_reference_pipeline.constants import WFI_REF_TYPES, WFI_TYPE_IMAGE
from astropy import units as u


class MakeDevMeta:
    """
    Class to generate any complete reference file MetaData object.

    Example Usage:
    from wfi_reference_pipeline.constants import REF_TYPE_DARK
    dev_meta_maker = MakeDevMeta(ref_type=REF_TYPE_DARK)
    dark_meta_data = dev_meta_maker.meta_dark

    """

    def _create_dev_meta_dark(self, meta_data):
        ngroups = 6
        nframes = 8
        groupgap = 0
        ma_table_name = "High Latitude Imaging Survey"
        ma_table_number = 1
        mode = WFI_MODE_WIM
        type = WFI_TYPE_IMAGE
        ref_optical_element = ["F158"]

        dark_meta_data = [ngroups, nframes, groupgap, ma_table_name, ma_table_number,
                          mode, type, ref_optical_element]
        self.meta_dark = WFIMetaDark(*meta_data, *dark_meta_data)

    def _create_dev_meta_flat(self, meta_data):
        p_optical_element = "F158"

        flat_meta_data = [p_optical_element]
        self.meta_flat = WFIMetaFlat(*meta_data, *flat_meta_data)

    def _create_dev_meta_gain(self, meta_data):
        self.meta_gain = WFIMetaGain(*meta_data)

    def _create_dev_meta_ipc(self, meta_data):
        p_optical_element = "F158"

        ipc_meta_data = [p_optical_element]
        self.meta_ipc = WFIMetaIPC(*meta_data, *ipc_meta_data)

    def _create_dev_meta_linearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        linearity_meta_data = [input_units, output_units]
        self.meta_linearity = WFIMetaLinearity(*meta_data,
                                               *linearity_meta_data)

    def _create_dev_meta_inverselinearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        inverselinearity_meta_data = [input_units, output_units]
        self.meta_inverselinearity = WFIMetaInverseLinearity(*meta_data,
                                                             *inverselinearity_meta_data)

    def _create_dev_meta_mask(self, meta_data):
        self.meta_mask = WFIMetaMask(*meta_data)

    def _create_dev_meta_readnoise(self, meta_data):
        mode = WFI_MODE_WIM
        type = WFI_TYPE_IMAGE

        readnoise_meta_data = [mode, type]
        self.meta_readnoise = WFIMetaReadNoise(*meta_data,
                                               *readnoise_meta_data)

    def _create_dev_meta_referencepixel(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        referencepixel_meta_data = [input_units, output_units]
        self.meta_referencepixel = WFIMetaReferencePixel(*meta_data,
                                                         *referencepixel_meta_data)

    def _create_dev_meta_saturation(self, meta_data):
        self.meta_saturation = WFIMetaSaturation(*meta_data)

    def __init__(self, ref_type):
        """
        Generates a reference type specific MetaData object relevant to the ref_type
        parameter.

        Parameters
        -------
        ref_type: str;
            String defining the reference file type which will determine the reference
            meta object created.
        """

        self.pedigree = "DUMMY"
        self.description = "For RFP Development and DMS Build Support."
        self.author = "RFP"
        self.use_after = "2023-01-01T00:00:00.000"
        self.telescope = "ROMAN"
        self.origin = "STSCI"
        self.instrument = "WFI"
        self.detector = "WFI01"

        if ref_type not in WFI_REF_TYPES:
            raise ValueError(f"ref_type must be one of: {WFI_REF_TYPES}")
        if self.pedigree not in WFI_PEDIGREE:
            raise ValueError(f"pedigree must be one of: {WFI_PEDIGREE}")
        if self.detector not in WFI_DETECTORS:
            raise ValueError(f"detector must be one of: {WFI_DETECTORS}")

        meta_data_params = [ref_type, self.pedigree, self.description, self.author,
                            self.use_after, self.telescope, self.origin, self.instrument, self.detector]

        if ref_type == "DARK":
            self._create_dev_meta_dark(meta_data_params)

        if ref_type == "FLAT":
            self._create_dev_meta_flat(meta_data_params)

        if ref_type == "GAIN":
            self._create_dev_meta_gain(meta_data_params)

        if ref_type == "INVERSELINEARITY":
            self._create_dev_meta_inverselinearity(meta_data_params)

        if ref_type == "IPC":
            self._create_dev_meta_ipc(meta_data_params)

        if ref_type == "LINEARITY":
            self._create_dev_meta_linearity(meta_data_params)

        if ref_type == "MASK":
            self._create_dev_meta_mask(meta_data_params)

        if ref_type == "READNOISE":
            self._create_dev_meta_readnoise(meta_data_params)

        if ref_type == "REFPIX":
            self._create_dev_meta_referencepixel(meta_data_params)

        if ref_type == "SATURATION":
            self._create_dev_meta_saturation(meta_data_params)
