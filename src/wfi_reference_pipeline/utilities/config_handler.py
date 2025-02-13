from pathlib import Path

import yaml
from jsonschema import exceptions, validate
from wfi_reference_pipeline.constants import (
    CONFIG_PATH,
    WFI_REF_TYPES,
    REF_TYPE_DARK,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.utilities.schemas import CONFIG_SCHEMA, QC_CONFIG_SCHEMA, CRDS_CONFIG_SCHEMA, PIPELINES_CONFIG_SCHEMA


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


def _validate_config(config_file_dict, schema):
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

    # Test that the provided config file dict matches the schema
    try:
        # Validate YAML data against the schema
        validate(instance=config_file_dict, schema=schema)
    except exceptions.ValidationError as e:
        raise exceptions.ValidationError(
            f"Provided config file dictionary does not match the required YML schema: {e}"
        )


def _get_config(config_filename):
    """Return a dictionary that holds the contents of config.yml

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """
    config_file_location = _find_config_file(config_filename)

    if config_file_location is None:
        raise FileNotFoundError(
            f"""The WFI_REFERENCE_PIPELINE package requires a {config_filename} inside
            the 'src/wfi_reference_pipeline/config' folder.
            """
        )

    with open(config_file_location, "r") as config_file:
        try:
            settings = yaml.safe_load(config_file)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Incorrectly formatted {config_filename}. Please fix YML formatting: {e}"
            )

    return settings


def get_logging_config(config_file="config.yml"):
    # Ensure the file has all the needed entries with expected data types
    settings = _get_config(config_file)
    _validate_config(settings, CONFIG_SCHEMA)
    return settings["logging"]


def get_data_files_config(config_file="config.yml"):
    settings = _get_config(config_file)
    _validate_config(settings, CONFIG_SCHEMA)
    return settings["data_files"]

def get_pipelines_config(ref_type, config_file="pipelines_config.yml"):
    settings = _get_config(config_file)
    _validate_config(settings, PIPELINES_CONFIG_SCHEMA)
    if ref_type == REF_TYPE_DARK:
        settings_type = "dark"
    else:
        raise ValueError(
            f"{ref_type} has not yet been implemented for configuration."
        )
    return settings[settings_type]


def get_quality_control_config(ref_type, config_file="quality_control_config.yml"):
    """Get configuration settings from quality_control_config.yml for any reference type
    Validate that the settings are in the correct format before returning
    """
    setting_type = ""
    try:
        if ref_type == REF_TYPE_DARK:
            settings = _get_config(config_file)
            setting_type = "dark_control"
        elif ref_type == REF_TYPE_READNOISE:
            settings = _get_config(config_file)["readnoise_control"]
            setting_type = "readnoise_control"
        else:
            raise ValueError(
                f"{ref_type} not a valid parameter.  Use one of the following: {list(WFI_REF_TYPES)}"
            )
    except KeyError as e:
        raise KeyError (f"Invalid schema index for ref_type: {ref_type} -- {e}")

    _validate_config(settings, QC_CONFIG_SCHEMA)
    return settings[setting_type]


def get_crds_submission_config(config_file="crds_submission_config.yml"):
    """
    Get configuration settings from crds_submission_config.yml.

    Parameters
    ----------
    config_file : str, optional
        The name of the configuration file to load, by default "crds_submission_config.yml".

    Returns
    -------
    dict
        A dictionary containing the configuration settings.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the YAML file is incorrectly formatted.
    ValidationError
        If the configuration does not match the CRDS_CONFIG_SCHEMA.
    """
    settings = _get_config(config_file)
    _validate_config(settings, CRDS_CONFIG_SCHEMA)
    return settings
