"""
Module for submitting reference files to CRDS. This will auto-populate
the information for the reference file submission.
"""

import os
import yaml
from crds.submit import Submission


def create_form(files, *args, yaml_form=None, submission_info=None,
                server='test', **kwargs):
    """
    Inputs
    ------
    files (str list):
        List of reference files that should be attached to the form
        for the delivery to CRDS.

    yaml_form (str; optional; default=None):
        The name of a YAML file containing the submission information
        for the form. If provided, this takes precedence over other
        inputs. Must provide either yaml_form OR submissino_info.

    submission_info (dict; optional; default=None):
        A dictionary containing the information for the submission form.
        Must provide either yaml_form OR submission_info.

    server (str; optional; default='test'):
        A string that specifies to which Roman CRDS server the reference
        files are being delivered. Must be one of either 'ops', 'test', or
        'dev'.

    Returns
    -------
    form (crds.submit.Submission):
        The submission form. This should be checked before using the form's
        submit() method.
    """

    # Check the inputs.
    if not isinstance(files, list):
        raise TypeError('Input files should be in a list.')
    if not yaml_form and not submission_info:
        raise ValueError('Must supply either yaml_form OR submission_info.')
    if server:
        if server.lower() not in ('ops', 'test', 'dev'):
            raise ValueError('If supplied, server must be one of ops, test, '
                             'or dev.')
    if len(files) == 0:
        raise ValueError('Input file list is empty!')
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f'File {file} could not be found! Is the '
                                    f'path correct?')

    # Save the inputs to attributes and set up remaining attributes.
    files = files
    if yaml_form:
        with open(yaml_form) as yf:
            submission_dict = yaml.safe_load(yf)
    else:
        submission_dict = submission_info
    form = Submission('roman', server.lower())

    # Update the form. We might be missing some keys that are present in
    # the form, and that's okay if they have default values or are not
    # required. This is up to the user to verify that it's correct and
    # complete for the delivery.
    for key in form.keys():
        try:
            form[key] = submission_dict[key]
        except KeyError:
            pass

    # Attach the reference files being delivered.
    for file in files:
        form.add_file(file)

    return form
