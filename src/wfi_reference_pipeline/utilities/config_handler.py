import os
from pathlib import Path

import yaml
from jsonschema import exceptions, validate
from wfi_reference_pipeline.constants import CONFIG_PATH


def _find_config_file(config_filename):
    """Find the config file using CONFIG path and
    our projects root directory

    Parameters
    ----------
    config_filename : string
        config file we're looking for within the CONFIG_PATH

    """
    current_path = Path.cwd()
    this_path = Path(current_path) / "pyproject.toml"
    if this_path.is_file():
        root_path = current_path
    else:
        # move up our directory path until we find pyproject.toml
        root_path = next(
            (path for path in current_path.parents if (path / "pyproject.toml").is_file()),
            None,
        )

    # once we have our root path, grab the path to the config_file
    config_file_path = root_path / CONFIG_PATH / config_filename
    if config_file_path.is_file():
        return config_file_path

    return None  # Config file not found


def _validate_config(config_file_dict):
    """Check that the config.yml file contains all the needed entries with
    expected data types

    Parameters
    ----------
    config_file_dict : dict
        The configuration YAML file loaded as a dictionary

    Notes
    -----
    See here for more information on JSON schemas:
        https://json-schema.org/learn/getting-started-step-by-step.html
    """
    # Define the schema for config.json
    schema = {
        "type": "object",
        "properties": {  # List all the possible entries and their types
            "Logging": {
                "type": "object",
                "properties": {
                    "log_dir": {"type": "string"},
                    "log_level": {"type": "string"},
                    "log_tag": {"type": "string"},
                },
                "required": ["log_dir", "log_level"],
            },
            "DataFiles": {
                "type": "object",
                "properties": {
                    "prep_dir": {"type": "string"},
                    "calibrated_dir": {"type": "string"},
                },
                "required": ["prep_dir", "calibrated_dir"],
            },
        },
        # List which entries are needed (all of them)
        "required": ["Logging", "DataFiles"],
    }

    # Test that the provided config file dict matches the schema
    try:
        # Validate YAML data against the schema
        validate(instance=config_file_dict, schema=schema)
    except exceptions.ValidationError as e:
        raise exceptions.ValidationError(
            f"Provided config.json does not match the required YML schema: {e}"
        )


def get_config():
    """Return a dictionary that holds the contents of config.yml

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """
    config_filename = "config.yml"
    if os.environ.get("READTHEDOCS") == "True":
        config_filename = "example_config.yml"

    config_file_location = _find_config_file(config_filename)

    if config_file_location is None:
        raise FileNotFoundError(
            """The WFI_REFERENCE_PIPELINE package requires a config.yml inside
            the 'src/wfi_reference_pipeline/config' folder.
            Use 'src/wfi_reference_pipeline/config/example_config.yml as a template
            """
        )

    with open(config_file_location, "r") as config_file:
        try:
            settings = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Incorrectly formatted config.yml file. Please fix YML formatting: {e}"
            )

    # Ensure the file has all the needed entries with expected data types
    _validate_config(settings)
    return settings


def get_logging_config():
    return get_config()["Logging"]

def get_datafiles_config():
    return get_config()["DataFiles"]
