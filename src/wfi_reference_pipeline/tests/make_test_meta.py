from wfi_reference_pipeline.resources.wfi_meta_dark import WFIMetaDark
from wfi_reference_pipeline.constants import WFI_REF_TYPES, WFI_PEDIGREE, WFI_DETECTORS


class MakeTestMeta:
    """
    Class to generate reference file common meta and type specific meta.
    """
    def _create_test_meta_dark(self, meta_data):
        ngroups = 1
        nframes = 1
        groupgap = 0
        ma_table_name = "Test ma_table_name"
        ma_table_number = 0
        p_optical_element = "F158"

        dark_meta_data = [ngroups, nframes, groupgap, ma_table_name, ma_table_number, p_optical_element]
        self.meta_dark = WFIMetaDark(*meta_data, *dark_meta_data)

    def __init__(self, ref_type=None):
        """
        Return common meta data on any instance of the class, with optional ref_type variable needed to import
        reference file specific meta for testing for files requiring additional information in meta as determined
        by the schema in roman data models.

        Parameters
        -------
        ref_type: str; default = None
            String defining the reference file type which will populate reftype in the common meta data for all
            reference files and necessary selector for additional meta.
        """

        pedigree = "TESTING"
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

        META_DATA_PARAMS = [WFI_REF_TYPES[ref_type], pedigree, description, author, use_after, telescope, origin, instrument, detector]

        if ref_type == "DARK":
            self._create_test_meta_dark(META_DATA_PARAMS)


