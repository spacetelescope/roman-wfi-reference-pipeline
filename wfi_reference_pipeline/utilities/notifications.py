"""
Tools for sending notifications from the pipeline.
"""

import importlib.resources as pkg_resources
import json
import logging
import os
import requests
import yaml

from wfi_reference_pipeline import config


def send_slack_message(message, token, config_file='slack_dev.yml'):
    """
    Function for sending messages to a slack channel.

    Inputs
    ------
    message (str):
        Message to send to the slack channel.

    config_file (str; optional; default="slack_dev.yml"):
        YAML file containing the configuration information for the slack
        integration including optional name/icon information.

    Returns
    -------
    None
    """

    try:
        # token = os.environ['WFI_SLACK_TOKEN']
        if config_file:
            with pkg_resources.open_text(config, config_file) as cf:
                slack_config = yaml.safe_load(cf)
        else:
            slack_config = {}

        data = json.dumps({'text': message, **slack_config})
        _ = requests.post(token, data=data)

    except KeyError:
        logging.warning('No slack token was provided, so skipping '
                        'the notification...')
