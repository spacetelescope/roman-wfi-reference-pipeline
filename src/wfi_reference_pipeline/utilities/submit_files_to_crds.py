"""
Module for submitting reference files to CRDS. This will auto-populate
the information for the reference file submission.
"""

import os
import yaml
from typing import Union
from dataclasses import dataclass
from crds.submit import Submission
from crds.certify import certify_files
from crds.core import heavy_client

from .notifications import send_slack_message
from .config_handler import get_data_files_config
from wfi_reference_pipeline.constants import WFI_REF_TYPES


@dataclass(init=True, repr=True)
class SubmissionForm:
    # deliverer: str = 'WFI Reference File Pipeline'  # Or specific user
    # other_email: str = 'rcosenti@stsci.edu'
    # instrument: str = 'WFI'
    # file_type: str = 'REF_TYPE'  # Needs to be a valid reference file
    # history_updated: bool = True
    # pedigree_updated: bool = True
    # keywords_checked: bool = True
    # descrip_updated: bool = True
    # useafter_updated: bool = True
    # useafter_matches: str = 'N/A'
    # compliance_verified: str = 'N/A'
    # etc_delivery: bool = False
    # calpipe_version: str = 'No'
    # replacement_files: bool = False
    # old_reference_files: Union[str, list] = 'List old files.'  # Needs to be updated by user
    # replacing_badfiles: str = 'No'
    # jira_issue: Union[str, list] = ''
    # table_rows_changed: str = 'N/A'
    # reprocess_affected: bool = False
    # modes_affected: str = 'WFI Imaging WIM and WFI Spectral WSM Modes'  # Should be checked by user
    # change_level: str = 'MODERATE'
    # correctness_testing: str = 'None'
    # additional_considerations: str = 'None'
    # description: str = "Intentionally left blank?"  # Needs to be updated by user.

    def __post_init__(self):

        self.fill_form_from_config()


        for list_meta in [self.old_reference_files, self.jira_issue]:
            if isinstance(list_meta, list):
                list_meta = ' '.join(list_meta)

        # Check if description has been updated by user
        if self.description == "Intentionally left blank?":
            raise ValueError('You did not update the reason for delivery. '
                             'Try again and set the description keyword!')

        # Check if file_type matches one of the WFI_REF_TYPES
        if self.file_type not in WFI_REF_TYPES:
            raise ValueError(f'The file_type "{self.file_type}" is not a valid WFI reference file type. '
                             f'Please select one of the following valid types: {", ".join(WFI_REF_TYPES)}')

        # Check if old_reference_files has been updated by user
        if self.old_reference_files == 'List old files.':
            raise ValueError('You did not update the list of old reference files. '
                             'Please provide the list of files being replaced.')

    def fill_submission_form_from_config(self):
        crds_submission_config = get_crds_submission_config()["submission_form"]
        self.deliverer = crds_submission_config["deliver"]
        self.other_email
        self.instrument
        self.file_type
        self.history_updated
        self.pedigree_updated: bool = True
        self.keywords_checked: bool = True
        self.descrip_updated: bool = True
        self.useafter_updated: bool = True
        self.useafter_matches: str = 'N/A'
        self.compliance_verified: str = 'N/A'
        self.etc_delivery: bool = False
        self.calpipe_version: str = 'No'
        self.replacement_files: bool = False
        self.old_reference_files: Union[str, list] = 'List old files.'  # Needs to be updated by user
        self.replacing_badfiles: str = 'No'
        self.jira_issue: Union[str, list] = ''
        self.table_rows_changed: str = 'N/A'
        self.reprocess_affected: bool = False
        self.modes_affected: str = 'WFI Imaging WIM and WFI Spectral WSM Modes'  # Should be checked by user
        self.change_level: str = 'MODERATE'
        self.correctness_testing: str = 'None'
        self.additional_considerations: str = 'None'
        self.description: str = "Intentionally left blank?"  # Needs to be updated by user.


    def as_dict(self) -> dict:
        return self.__dict__


class WFISubmit:

    def __init__(self, files, submission_info=None, server='test'):
        """
        Initialize the submission process with file checks and configuration.
        """
        if isinstance(files, list) or isinstance(files, tuple):
            if len(files) > 0:
                self.files = files
            else:
                raise ValueError(f'Input files list/tuple is empty! Got {files}.')
        else:
            raise TypeError(f'Input files should be a list or tuple. Got {type(files)} instead.')

        for f in self.files:
            if not os.path.exists(f):
                raise FileNotFoundError(f'Input file {f} does not exist!')

        self.server = server.lower()
        if self.server not in ('dev', 'test', 'tvac'):
            raise ValueError(f'Server should be either "test" or "dev" or "tvac". Got {self.server} instead.')

        if submission_info is None:
            # Load submission information from config file if not provided
            submission_info = get_data_files_config()['submission_form']

        if isinstance(submission_info, str):
            with open(submission_info, 'r') as subfile:
                self.submission_dict = yaml.safe_load(subfile)
        elif isinstance(submission_info, dict):
            self.submission_dict = submission_info
        else:
            raise TypeError(f'submission_info should be either a dictionary or '
                            f'the string name of a YAML file. Got {type(submission_info)} instead.')

        self.submission_form = Submission('roman', self.server)
        self.submission_results = None


    def get_form_keys(self):
        """
        Print available keys for the submission form.
        """
        print(self.submission_form.help())

    def certify_reffiles(self):
        """
        Certify the reference files before submission.
        """
        cert_files = [f if '/' in f else f'./{f}' for f in self.files]
        server_info = heavy_client.get_config_info('roman')
        context = server_info['operational_context']

        certify_files(cert_files, context)

    def update_form_with_config(self):
        """
        Update the submission form with the provided configuration.
        """
        for key in self.submission_form.keys():
            try:
                self.submission_form[key] = self.submission_dict[key]
            except KeyError:
                pass

        # Attach the reference files being delivered.
        for f in self.files:
            self.submission_form.add_file(f)

    def submit_to_crds(self, summary=None):
        """
        Submit the files to CRDS. This is the actual submission to CRDS method.

        Parameters
        ----------
        summary: str; default = None
            Message to send to the slack channel.
        """
        self.submission_results = self.submission_form.submit()
        try:
            if summary:
                summary += f' Result URL is {self.submission_results.ready_url}'
            else:
                summary = 'Files have been submitted. No summary provided.'
            send_slack_message(summary, os.environ['WFI_REFFILE_SLACK_TOKEN'],
                               config_file=None)
        except KeyError:
            raise Exception('No token found in environment variable "WFI_REFFILE_SLACK_TOKEN". '
                            'No Slack message was sent.')
