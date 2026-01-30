from pathlib import Path

import yaml
from jsonschema import exceptions, validate

from wfi_reference_pipeline.utilities.schemas import (
    CONFIG_SCHEMA,
    CRDS_CONFIG_SCHEMA,
    PIPELINES_CONFIG_SCHEMA,
    QC_CONFIG_SCHEMA,
)


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
    #TODO - Schema check not currently working (as in everything passes. BAD!)
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

     Parameters
    ----------
    config_file : str, optional
        The name of the configuration file to load, by default "crds_submission_config.yml".

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """

    # all config files stored in same directory
    current_path = Path(__file__).parent.resolve()
    config_file_location = current_path / config_filename
    if not config_file_location.is_file():
        config_file_location = None  # Config file not found

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
    """Get configuration settings from config.yml for logging
    Validate that the settings are in the correct format before returning

    Parameters
    ----------
    config_file : str, optional
        The name of the configuration file to load, by default "config.yml".

    Returns
    -------
    dict
        A dictionary containing the configuration settings.
    """
    # Ensure the file has all the needed entries with expected data types
    settings = _get_config(config_file)
    _validate_config(settings, CONFIG_SCHEMA)
    return settings["logging"]


def get_data_files_config(config_file="config.yml"):
    """Get configuration settings from config.yml for data_files
    Validate that the settings are in the correct format before returning

    Parameters
    ----------
    config_file : str, optional
        The name of the configuration file to load, by default "config.yml".

    Returns
    -------
    dict
        A dictionary containing the configuration settings.
    """
    settings = _get_config(config_file)
    _validate_config(settings, CONFIG_SCHEMA)
    return settings["data_files"]

def get_db_config(config_file="config.yml"):
    """Get configuration settings from config.yml for database
    Validate that the settings are in the correct format before returning

    Parameters
    ----------
    config_file : str, optional
        The name of the configuration file to load, by default config.yml.

    Returns
    -------
    dict
        A dictionary containing the configuration settings.
    """
    settings = _get_config(config_file)
    _validate_config(settings, CONFIG_SCHEMA)
    return settings["database"]


def get_pipelines_config(ref_type, config_file="pipelines_config.yml"):
    """Get configuration settings from pipelines_config.yml for any reference type
    Validate that the settings are in the correct format before returning

    Parameters
    ----------
    ref_type : CONSTANT
        The defined reference type from constants.py
    config_file : str, optional
        The name of the configuration file to load, by default "pipelines_config.yml".

    Returns
    -------
    dict
        A dictionary containing the configuration settings.

    Raises
    ------
    KeyError
        If the ref_type hasn't been implemented yet
    """
    settings = _get_config(config_file)
    _validate_config(settings, PIPELINES_CONFIG_SCHEMA)
    try:
        ref_type_config = settings[ref_type.lower()]
    except KeyError:
        raise KeyError(f"{ref_type.lower()} has not yet been implemented in {config_file}.")
    return ref_type_config


def get_quality_control_config(ref_type, detector=None, config_file=None):
    """Get configuration settings from quality_control_config.yml for any reference type
    Validate that the settings are in the correct format before returning

    Parameters
    ----------
    ref_type : CONSTANT
        The defined reference type from constants.py
    config_file : str, optional
        The name of the configuration file to load, by default None.

    Returns
    -------
    dict
        A dictionary containing the configuration settings.

    Raises
    ------
    ValueError
        If the YAML file is incorrectly formatted.
    KeyError
        If the configuration can't find your ref type.
    """
    if config_file is None:
        if detector is None:
            raise ValueError("Must send in detector or config_file for Quality Control")
        config_file = f"quality_control_config_{detector}.yml"
    settings = _get_config(config_file)
    _validate_config(settings, QC_CONFIG_SCHEMA)
    try:
        ref_type_config = settings[ref_type.lower()]
    except KeyError:
        raise KeyError(f"{ref_type.lower()} has not yet been implemented in {config_file}.")
    return ref_type_config


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
