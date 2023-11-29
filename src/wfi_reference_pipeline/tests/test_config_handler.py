import os
import pytest

from wfi_reference_pipeline.utilities.config_handler import get_config

# Determine if tests are being run on Gitlab CI
ON_GITHUB_ACTIONS = "CI" in os.environ


@pytest.mark.skipif(ON_GITHUB_ACTIONS, reason="CI tests don't have access to config.yml")
def test_get_config():
    """Assert that the ``get_config`` function successfully creates a dictionary."""
    settings = get_config()
    assert isinstance(settings, dict)
