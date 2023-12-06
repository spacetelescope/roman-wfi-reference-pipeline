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


@dataclass(init=True, repr=True)
class SubmissionForm:
    deliverer: str = 'WFI Reference File Pipeline'
    other_email: str = ''
    instrument: str = 'WFI'
    file_type: str = 'Blank'
    history_updated: bool = True
    pedigree_updated: bool = True
    keywords_checked: bool = True
    descrip_updated: bool = True
    useafter_updated: bool = True
    useafter_matches: str = 'N/A'
    compliance_verified: str = 'N/A'
    etc_delivery: bool = False
    calpipe_version: str = 'No'
    replacement_files: bool = False
    old_reference_files: Union[str, list] = ''
    replacing_badfiles: str = 'No'
    jira_issue: Union[str, list] = ''
    table_rows_changed: str = 'N/A'
    reprocess_affected: bool = False
    modes_affected: str = 'All WFI modes'
    change_level: str = 'MODERATE'
    correctness_testing: str = 'None'
    additional_considerations: str = 'None'
    description: str = "You didn't update this, did you?"

    def __post_init__(self):

        for list_meta in [self.old_reference_files, self.jira_issue]:
            if isinstance(list_meta, list):
                list_meta = ' '.join(list_meta)

        if self.description == "You didn't update this, did you?":
            raise ValueError('You did not update the reason for delivery. Try again and set the description keyword to something useful!')

        if self.file_type == 'Blank':
            raise ValueError('You did not update the reference file type. Try again and set the file_type keyword to the correct values!')

    def as_dict(self) -> dict:
        return self.__dict__


class WFISubmit:

    def __init__(self, files, submission_info, server='test'):

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
        if self.server not in ('dev', 'test'):
            raise ValueError(f'server should be either "test" or "dev". Got {self.server} instead.')

        if isinstance(submission_info, str):
            with open(submission_info, 'r') as subfile:
                self.submission_dict = yaml.safe_load(subfile)
        elif isinstance(submission_info, dict):
            self.submission_dict = submission_info
        else:
            raise TypeError(f'submission_info should be either a dictionary or the string name of a YAML file. Got {type(submission_info)} instead.')

        self.submission_form = Submission('roman', self.server)
        self.submission_results = None

    def get_form_keys(self):

        print(self.submission_form.help())

    def certify_reffiles(self):

        cert_files = [f if '/' in f else f'./{f}' for f in self.files]
        server_info = heavy_client.get_config_info('roman')
        context = server_info['operational_context']

        certify_files(cert_files, context)

    def update_form(self):

        for key in self.submission_form.keys():
            try:
                self.submission_form[key] = self.submission_dict[key]
            except KeyError:
                pass

        # Attach the reference files being delivered.
        for f in self.files:
            self.submission_form.add_file(f)

    def submit_to_crds(self, summary=None):

        # not sure if we want this flag, but it was in the old submit_files.py
        #self.submission_form._auto_confirm = True
        self.submission_results = self.submission_form.submit()
        try:
            if summary:
                summary = summary
                summary += f' Result URL is {self.submission_results.ready_url}'
            else:
                summary = 'Files have been submitted. No summary provided.'
                send_slack_message(summary, os.environ['WFI_REFFILE_SLACK_TOKEN'],
                                   config_file=None)

        except KeyError:
            raise Exception('No token found in environment variable "WFI_REFFILE_SLACK_TOKEN". No Slack message was sent.')
