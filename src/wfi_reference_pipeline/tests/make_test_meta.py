import yaml
from pathlib import Path


class MakeTestMeta:
    """
    Class to generate reference file common meta and type specific meta required to validate against schema in roman
    data models.
    """

    def __init__(self, ref_type=None):
        """
        Return common meta data on any instance of the class, with optional ref_type variable needed to import
        reference file specific meta for testing for files requiring additional information in meta as determined
        by the schema in roman data models.

        Parameters
        -------
        ref_type: str; default = None
            String defining the reference file type which will populate additional meta data for file types requiring
            meta not included in common.

        """

        # File to dictionary with empty common meta.
        common_yaml_path = Path("/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/src/wfi_reference_pipeline/"
                                "resources/data/common_meta.yaml")

        # Load the YAML file contents into a dictionary using safe_load().
        with common_yaml_path.open() as cyf:
            self.test_meta = yaml.safe_load(cyf)
            self.test_meta.update({'reftype': ref_type})
            self.test_meta.update({'pedigree': 'DUMMY'})
            self.test_meta.update({'description': 'For RFP testing.'})
            self.test_meta.update({'author': 'RFP Test Suite'})
            self.test_meta.update({'useafter': '2020-01-01T00:00:00.000'})
            self.test_meta.update({'telescope': 'ROMAN'})
            self.test_meta.update({'origin': 'STSCI'})
            self.test_meta['instrument'].update({'name': 'WFI'})
            self.test_meta['instrument'].update({'detector': 'WFI01'})

            # Add Dark and Read Noise required exposure data about type.
            if self.test_meta['reftype'] in ['DARK', 'READ_NOISE']:
                self.test_meta['exposure'] = {'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'}
                # Add Dark specific exposure meta about MA table.
                if self.test_meta['reftype'] == 'DARK':
                    self.test_meta['exposure'].update({'ngroups': 1, 'nframes': 1, 'groupgap': 0,
                                                       'ma_table_name': 'Test Table', 'ma_table_number': 0})

            # Add Dark, Distortion, Flat, and IPC required optical element data about filter.
            if self.test_meta['reftype'] in ['DARK', 'DISTORTION', 'FLAT', 'IPC']:
                self.test_meta['instrument'].update({'optical_element': 'F158'})
