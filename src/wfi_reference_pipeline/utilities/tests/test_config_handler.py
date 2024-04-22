import os
import pytest

from wfi_reference_pipeline.utilities.config_handler import get_config, get_logging_config

# Determine if tests are being run on Gitlab CI
IN_CI_ENVIRONMENT = "CI" in os.environ


@pytest.mark.skipif(IN_CI_ENVIRONMENT, reason="CI tests don't have access to config.yml")
def test_get_config():
    """Assert that the ``get_config`` function successfully creates a dictionary."""
    settings = get_config()
    assert isinstance(settings, dict)

@pytest.mark.skipif(IN_CI_ENVIRONMENT, reason="CI tests don't have access to config.yml")
def test_get_logging_config():
    """Assert that the ``get_config`` function successfully creates a dictionary."""
    settings = get_logging_config()
    assert isinstance(settings, dict)