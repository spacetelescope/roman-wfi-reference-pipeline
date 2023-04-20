import yaml, importlib.resources
import wfi_reference_pipeline.resources.data as resource_meta


class MakeTestMeta:
    """
    Class to generate reference file common meta and type specific meta.
    """
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

        # File to empty dictionary of common meta keys.
        meta_yaml_files = importlib.resources.files(resource_meta)

        # Load the YAML file contents into a dictionary using safe_load().
        common_yaml_path = meta_yaml_files.joinpath('common_meta.yaml')
        with common_yaml_path.open() as cyp:
            self.test_meta = yaml.safe_load(cyp)
            self.test_meta.update({'reftype': ref_type})
            self.test_meta.update({'pedigree': 'DUMMY'})
            self.test_meta.update({'description': 'For RFP testing.'})
            self.test_meta.update({'author': 'RFP Test Suite'})
            self.test_meta.update({'useafter': '2020-01-01T00:00:00.000'})
            self.test_meta.update({'telescope': 'ROMAN'})
            self.test_meta.update({'origin': 'STSCI'})
            self.test_meta['instrument'].update({'name': 'WFI'})
            self.test_meta['instrument'].update({'detector': 'WFI01'})

            # Add Dark and Read Noise exposure type meta.
            if self.test_meta['reftype'] in ['DARK', 'READ_NOISE']:
                self.test_meta['exposure'] = {'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'}
                # Add Dark MA table meta.
                if self.test_meta['reftype'] == 'DARK':
                    self.test_meta['exposure'].update({'ngroups': 1, 'nframes': 1, 'groupgap': 0,
                                                       'ma_table_name': 'Test Table', 'ma_table_number': 0})

            # Add optical element filter meta.
            if self.test_meta['reftype'] in ['DARK', 'DISTORTION', 'FLAT', 'IPC']:
                self.test_meta['instrument'].update({'optical_element': 'F158'})
                if self.test_meta['reftype'] in ['IPC']:
                    self.test_meta['instrument'].update({'p_optical_element':
                                                         'F062|F087|F106|F129|F146|F158|F184|F213|GRISM|PRISM|DARK|'})
