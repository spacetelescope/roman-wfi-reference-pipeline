"""
Tools for sending notifications from the pipeline.
"""

import importlib.resources as pkg_resources
import json
import requests
import yaml

from wfi_reference_pipeline import config


def send_slack_message(message, config_file='slack_dev.yml'):
    """
    Function for sending messages to a slack channel.

    Inputs
    ------
    message (str): Message to send to the slack channel.
    config_file (str; optional; default="slack_dev.yml"): YAML file containing
        the configuration information for the slack integration including
        the webhook URL and optional name/icon information.

    Returns
    -------
    None
    """

    with pkg_resources.open_text(config, config_file) as cf:
        slack_config = yaml.safe_load(cf)

    data = json.dumps({'text': message, **slack_config['bot_info']})
    _ = requests.post(slack_config['url'], data=data)
