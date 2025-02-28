import pytest
from wfi_reference_pipeline.constants import REF_TYPE_DARK
from wfi_reference_pipeline.config.config_handler import get_data_files_config, get_logging_config, get_quality_control_config

def test_bad_filename_fail_with_file_not_found_error():
    """Assert that the if a bad file name is sent, we get the correct error."""
    with pytest.raises(FileNotFoundError):
        get_logging_config("bad_example_config.yml")

def test_get_logging_config_pass():
    """Assert that the ``get_logging_config`` function successfully creates a dictionary."""
    settings = get_logging_config("example_config.yml")
    assert isinstance(settings, dict)

def test_get_data_files_config_pass():
    """Assert that the ``get_data_files_config`` function successfully creates a dictionary."""
    settings = get_data_files_config("example_config.yml")
    assert isinstance(settings, dict)


def test_get_quality_control_config():
    """Assert that the ``get_quality_control_config`` function successfully creates a dictionary."""
    settings = get_quality_control_config(REF_TYPE_DARK)
    assert isinstance(settings, dict)