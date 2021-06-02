# import json
import os
# import requests
import yaml

from astropy.utils.data import get_pkg_data_filename


def send_slack_message(message,
                       config_file=os.path.join('config', 'slack.yml')):

    with open(get_pkg_data_filename(config_file)) as config:
        slack_config = yaml.safe_load(config)

    print(message, slack_config)
