import importlib.resources as pkg_resources
import json
import requests
import yaml

from wfi_reference_pipeline import config


def send_slack_message(message, config_file='slack_devtest.yml'):

    with pkg_resources.open_text(config, config_file) as cf:
        slack_config = yaml.safe_load(cf)

    data = json.dumps({'text': message, **slack_config['bot_info']})
    _ = requests.post(slack_config['url'], data=data)
