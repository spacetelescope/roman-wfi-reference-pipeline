from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark
from wfi_reference_pipeline.resources.wfi_meta_inverselinearity import WFIMetaInverseLinearity
from wfi_reference_pipeline.resources.wfi_meta_referencepixel import WFIMetaReferencePixel
from wfi_reference_pipeline.utilities.wfi_meta_ipc import WFIMetaIPC
from wfi_reference_pipeline.resources.wfi_meta_linearity import WFIMetaLinearity
from wfi_reference_pipeline.constants import WFI_DETECTORS, WFI_MODE_WIM, WFI_PEDIGREE
from wfi_reference_pipeline.constants import WFI_REF_TYPES, WFI_TYPE_IMAGE
from astropy import units as u


class MakeTestMeta:
    """
    Class to generate any complete reference file MetaData object.

    Example Usage:
    test_meta_maker = MakeTestMeta("DARK")
    dark_meta_data = test_meta_maker.meta_dark

    """
    def _create_test_meta_ipc(self, meta_data):
        ref_optical_element = "F158"

        ipc_meta_data = [ref_optical_element]
        self.meta_ipc = WFIMetaIPC(*meta_data, *ipc_meta_data)

    def _create_test_meta_inverselinearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        inverselinearity_meta_data = [input_units, output_units]
        self.meta_inverselinearity = WFIMetaInverseLinearity(*meta_data,
                                                             *inverselinearity_meta_data)

    def _create_test_meta_referencepixel(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        referencepixel_meta_data = [input_units, output_units]
        self.meta_referencepixel = WFIMetaReferencePixel(*meta_data,
                                                         *referencepixel_meta_data)

    def _create_test_meta_dark(self, meta_data):
        ngroups = 1
        nframes = 1
        groupgap = 0
        ma_table_name = "Test ma_table_name"
        ma_table_number = 0
        mode = WFI_MODE_WIM
        type = WFI_TYPE_IMAGE
        ref_optical_element = ["F158"]

        dark_meta_data = [ngroups, nframes, groupgap, ma_table_name, ma_table_number,
                          mode, type, ref_optical_element]
        self.meta_dark = WFIMetaDark(*meta_data, *dark_meta_data)

    def _create_test_meta_linearity(self, meta_data):
        input_units = u.DN
        output_units = u.DN

        linearity_meta_data = [input_units, output_units]
        self.meta_linearity = WFIMetaLinearity(*meta_data, *linearity_meta_data)

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

        pedigree = "DUMMY"
        description = "For RFP testing."
        author = "RFP Test Suite"
        use_after = "2023-01-01T00:00:00.000"
        telescope = "ROMAN"
        origin = "STSCI"
        instrument = "WFI"
        detector = "WFI01"

        if ref_type not in WFI_REF_TYPES:
            raise ValueError(f"ref_type must be one of: {WFI_REF_TYPES}")
        if pedigree not in WFI_PEDIGREE:
            raise ValueError(f"pedigree must be one of: {WFI_PEDIGREE}")
        if detector not in WFI_DETECTORS:
            raise ValueError(f"detector must be one of: {WFI_DETECTORS}")

        META_DATA_PARAMS = [WFI_REF_TYPES[ref_type], pedigree, description, author,
                            use_after, telescope, origin, instrument, detector]

        if ref_type == "DARK":
            self._create_test_meta_dark(META_DATA_PARAMS)

        if ref_type == "INVERSELINEARITY":
            self._create_test_meta_inverselinearity(META_DATA_PARAMS)

        if ref_type == "IPC":
            self._create_test_meta_ipc(META_DATA_PARAMS)

        if ref_type == "REFPIX":
            self._create_test_meta_referencepixel(META_DATA_PARAMS)

        if ref_type == "LINEARITY":
            self._create_test_meta_referencepixel(META_DATA_PARAMS)
