import os
import yaml
from jsonschema import validate, exceptions
from pathlib import Path


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
                },
                "required": ["log_dir", "log_level"],
            },
        },
        # List which entries are needed (all of them)
        "required": ["Logging"],
    }

    # Test that the provided config file dict matches the schema
    try:
        # Validate YAML data against the schema
        validate(instance=config_file_dict, schema=schema)
        print("YAML data is valid against the schema.")
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
    config_file_name = "config.yml"
    if os.environ.get("READTHEDOCS") == "True":
        config_file_name = "example_config.yml"

    config_file_location = (Path.cwd().parent / 'config' / config_file_name).resolve()

    if not config_file_location.exists():
        raise FileNotFoundError(
            "The WFI_REFERENCE_PIPELINE package requires a config.yml inside the 'src/wfi_reference_pipeline' folder"
        )

    with open(config_file_location, "r") as config_file:
        try:
            # load the yaml
            settings = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Incorrectly formatted config.yml file. Please fix YML formatting: {e}"
            )

    # Ensure the file has all the needed entries with expected data types
    _validate_config(settings)
    return settings
