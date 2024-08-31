"""
Module for submitting reference files to CRDS. This will auto-populate
the information for the reference file submission.
"""

import os
from typing import Union
from dataclasses import dataclass, field
from crds.submit import Submission
from crds.certify import certify_files
from crds.core import heavy_client

from .notifications import send_slack_message
from .config_handler import get_crds_submission_config
from wfi_reference_pipeline.constants import WFI_REF_TYPES


@dataclass(init=True, repr=True)
class SubmissionForm:
    deliverer: str = field(default='WFI Reference File Pipeline')
    other_email: str = field(default='')
    instrument: str = field(default='WFI')
    file_type: str = field(default='REF_TYPE')
    history_updated: bool = field(default=True)
    pedigree_updated: bool = field(default=True)
    keywords_checked: bool = field(default=True)
    descrip_updated: bool = field(default=True)
    useafter_updated: bool = field(default=True)
    useafter_matches: str = field(default='N/A')
    compliance_verified: str = field(default='N/A')
    etc_delivery: bool = field(default=False)
    calpipe_version: str = field(default='No')
    replacement_files: bool = field(default=False)
    old_reference_files: Union[str, list] = field(default='')
    replacing_badfiles: str = field(default='No')
    jira_issue: Union[str, list] = field(default='')
    table_rows_changed: str = field(default='N/A')
    reprocess_affected: bool = field(default=False)
    modes_affected: str = field(default='')
    change_level: str = field(default='MODERATE')
    correctness_testing: str = field(default='None')
    additional_considerations: str = field(default='None')
    description: str = field(default="")

    def __post_init__(self):

        self.fill_submission_form_with_crds_config()

    def fill_submission_form_with_crds_config(self):
        crds_submission_config = get_crds_submission_config()["submission_form"]
        self.deliverer = crds_submission_config.get("deliverer", self.deliverer)
        self.other_email = crds_submission_config.get("other_email", self.other_email)
        self.file_type = crds_submission_config.get("file_type", self.file_type)
        self.history_updated = crds_submission_config.get("history_updated", self.history_updated)
        self.pedigree_updated = crds_submission_config.get("pedigree_updated", self.pedigree_updated)
        self.keywords_checked = crds_submission_config.get("keywords_checked", self.keywords_checked)
        self.descrip_updated = crds_submission_config.get("descrip_updated", self.descrip_updated)
        self.useafter_updated = crds_submission_config.get("useafter_updated", self.useafter_updated)
        self.useafter_matches = crds_submission_config.get("useafter_matches", self.useafter_matches)
        self.compliance_verified = crds_submission_config.get("compliance_verified", self.compliance_verified)
        self.etc_delivery = crds_submission_config.get("etc_delivery", self.etc_delivery)
        self.calpipe_version = crds_submission_config.get("calpipe_version", self.calpipe_version)
        self.replacement_files = crds_submission_config.get("replacement_files", self.replacement_files)
        self.old_reference_files = crds_submission_config.get("old_reference_files", self.old_reference_files)
        self.replacing_badfiles = crds_submission_config.get("replacing_badfiles", self.replacing_badfiles)
        self.jira_issue = crds_submission_config.get("jira_issue", self.jira_issue)
        self.table_rows_changed = crds_submission_config.get("table_rows_changed", self.table_rows_changed)
        self.reprocess_affected = crds_submission_config.get("reprocess_affected", self.reprocess_affected)
        self.modes_affected = crds_submission_config.get("modes_affected", self.modes_affected)
        self.change_level = crds_submission_config.get("change_level", self.change_level)
        self.correctness_testing = crds_submission_config.get("correctness_testing", self.correctness_testing)
        self.additional_considerations = crds_submission_config.get("additional_considerations",
                                                                    self.additional_considerations)
        self.description = crds_submission_config.get("description", self.description)

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
        if self.server not in ('test', 'tvac', 'ops'):
            raise ValueError(f'Server should be either "test" or "tvac" or "ops". Got {self.server} instead.')

        if submission_info is None:
            # Load submission information from the config file if not provided
            self.submission_dict = get_crds_submission_config()['submission_form']
        else:
            self.submission_dict = submission_info

        # Use CRDS Submission to create submission_form
        self.submission_file_ref_type = None
        self.crds_submission_form = Submission('roman', self.server)
        self.submission_results = None

    def get_form_keys(self):
        """
        Print available keys for the submission form.
        """
        print(self.crds_submission_form.help())

    def certify_reffiles(self):
        """
        Certify the reference files before submission.
        """
        cert_files = [f if '/' in f else f'./{f}' for f in self.files]
        server_info = heavy_client.get_config_info('roman')
        context = server_info['operational_context']

        certify_files(cert_files, context)

    def update_crds_submission_form(self):
        """
        Update the crds submission form with the provided configuration
        and files.
        """

        self._check_for_default_strings()
        self._confirm_submission_file_type_is_rfp_ref_type()
        for key in self.crds_submission_form.keys():
            try:
                self.crds_submission_form[key] = self.submission_dict[key]
            except KeyError:
                pass

        # Attach the reference files being delivered.
        for f in self.files:
            self.crds_submission_form.add_file(f)
            self._confirm_ref_type_files_match_submission()

    def _confirm_submission_file_type_is_rfp_ref_type(self):
        """
        Check that the submitted file type is one
        of the rfp ref types.
        """

        if self.submission_dict['file_type'] not in WFI_REF_TYPES:
            raise ValueError(
                f"The file type '{self.submission_dict['file_type']}' is not one of the following: "
                f"{', '.join(WFI_REF_TYPES)}"
            )
        self.submission_file_ref_type = self.submission_dict['file_type']

    def _check_for_default_strings(self):
        """
        Check if any string fields have the placeholder value 'User updated information.'
        """
        placeholders = []
        for key, value in self.submission_dict.items():
            if isinstance(value, str) and value == 'User updated information.':
                placeholders.append(key)
        if placeholders:
            raise ValueError(
                f"The following have not been updated from their default value: {', '.join(placeholders)}")
        else:
            print("All fields are correctly updated.")

    def _confirm_ref_type_files_match_submission(self):
        """
        # TODO
        Write code to open each file that is being add and check the ref_type matches
        the reference file type in the submission form.
        """

        pass

    def submit_to_crds(self, summary=None):
        """
        Submit the files to CRDS. This is the actual submission to CRDS method.

        Parameters
        ----------
        summary: str; default = None
            Message to send to the slack channel.
        """
        self.submission_results = self.crds_submission_form.submit()
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
