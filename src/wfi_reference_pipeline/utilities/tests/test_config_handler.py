from wfi_reference_pipeline.utilities.config_handler import get_config, get_logging_config

def test_get_config():
    """Assert that the ``get_config`` function successfully creates a dictionary."""
    settings = get_config("example_config.yml")
    assert isinstance(settings, dict)

def test_get_logging_config():
    """Assert that the ``get_config`` function successfully creates a dictionary."""
    settings = get_logging_config("example_config.yml")
    assert isinstance(settings, dict)