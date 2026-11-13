import copy

import pytest  # noqa
from jsonschema import exceptions

from wfi_reference_pipeline.config.config_access import (  # noqa
    _validate_config,
    get_crds_config,
    get_crds_handler_config,
    get_crds_submission_config,
    get_data_files_config,
    get_logging_config,
    get_quality_control_config,
)
from wfi_reference_pipeline.constants import REF_TYPE_DARK  # noqa
from wfi_reference_pipeline.utilities.schemas import CRDS_CONFIG_SCHEMA


def test_bad_filename_fail_with_file_not_found_error():
    """Assert that the if a bad file name is sent, we get the correct error."""
    with pytest.raises(FileNotFoundError):
        get_logging_config("bad_example_config.yml")

# TODO this stopped working in the CI.  Evaluate and fix, but not part of this MR

# def test_get_logging_config_pass():
#     """Assert that the ``get_logging_config`` function successfully creates a dictionary."""
#     settings = get_logging_config("example_config.yml")
#     assert isinstance(settings, dict)

# def test_get_data_files_config_pass():
#     """Assert that the ``get_data_files_config`` function successfully creates a dictionary."""
#     settings = get_data_files_config("example_config.yml")
#     assert isinstance(settings, dict)


# def test_get_quality_control_config():
#     """Assert that the ``get_quality_control_config`` function successfully creates a dictionary."""
#     settings = get_quality_control_config(REF_TYPE_DARK)
#     assert isinstance(settings, dict)

# -------------------------------
# crds_config.yml
# -------------------------------
def test_get_crds_config_has_both_sections():
    """The shipped crds_config.yml validates and has both sections."""
    settings = get_crds_config()
    assert set(settings) == {"crds_handler", "submission_form"}


def test_get_crds_handler_config_defaults():
    """The crds_handler section holds the expected defaults."""
    settings = get_crds_handler_config()
    assert settings == {
        "server_url": None,
        "crds_path": None,
        "crds_context": None,
        "context_state": "latest",
        "exposure_type": "WFI_IMAGE",
        "sync_mappings": True,
    }


def test_get_crds_submission_config():
    """The submission_form section is returned on its own, without a wrapper key."""
    settings = get_crds_submission_config()
    assert settings["instrument"] == "WFI"
    assert settings["change_level"] == "MODERATE"
    assert "submission_form" not in settings


@pytest.mark.parametrize(
    "section, setting, value",
    [
        ("crds_handler", "exposure_type", "WFI_BOGUS"),
        ("crds_handler", "crds_context", "jwst_0042.pmap"),
        ("crds_handler", "crds_context", "roman_42.pmap"),
        ("crds_handler", "server_url", "roman-crds.stsci.edu"),
        ("crds_handler", "sync_mappings", "yes"),
        ("crds_handler", "misspelled_setting", True),
        ("submission_form", "etc_delivery", "no"),
    ],
)
def test_crds_config_schema_rejects_bad_values(section, setting, value):
    """Bad values and unknown crds_handler settings fail schema validation."""
    settings = copy.deepcopy(get_crds_config())
    settings[section][setting] = value
    with pytest.raises(exceptions.ValidationError):
        _validate_config(settings, CRDS_CONFIG_SCHEMA, "test")


@pytest.mark.parametrize("section", ["crds_handler", "submission_form"])
def test_crds_config_schema_requires_both_sections(section):
    """Leaving out either section fails schema validation."""
    settings = copy.deepcopy(get_crds_config())
    del settings[section]
    with pytest.raises(exceptions.ValidationError):
        _validate_config(settings, CRDS_CONFIG_SCHEMA, "test")


def test_crds_config_schema_accepts_pinned_context():
    """A specific Roman context and a server URL are valid settings."""
    settings = copy.deepcopy(get_crds_config())
    settings["crds_handler"]["crds_context"] = "roman_0042.pmap"
    settings["crds_handler"]["server_url"] = "https://roman-crds.stsci.edu"
    _validate_config(settings, CRDS_CONFIG_SCHEMA, "test")
