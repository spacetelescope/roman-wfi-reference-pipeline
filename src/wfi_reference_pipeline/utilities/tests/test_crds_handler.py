"""
Tests for how CrdsHandler applies its config. CRDS is monkeypatched, so no
server or CRDS cache is needed.
"""
import pytest

from wfi_reference_pipeline.constants import WFI_TYPE_GRISM, WFI_TYPE_IMAGE
from wfi_reference_pipeline.utilities import crds_handler as crds_handler_module
from wfi_reference_pipeline.utilities.crds_handler import CrdsHandler

DEFAULT_CONTEXT = "roman_0099.pmap"


@pytest.fixture
def config(monkeypatch):
    """
    Shipped config values, returned by a patched get_crds_handler_config so each
    test can change them. Also patches the server lookup for the default context.
    """
    settings = {
        "server_url": None,
        "crds_path": None,
        "crds_context": None,
        "context_state": "latest",
        "exposure_type": WFI_TYPE_IMAGE,
        "sync_mappings": True,
    }
    monkeypatch.setattr(crds_handler_module, "get_crds_handler_config", lambda config_file: settings)

    def fake_default_context(observatory, state):
        settings["_requested_state"] = state
        return DEFAULT_CONTEXT

    monkeypatch.setattr(crds_handler_module.crds, "get_default_context", fake_default_context)
    return settings


def test_defaults_come_from_config(config):
    handler = CrdsHandler()
    assert handler.crds_context == DEFAULT_CONTEXT
    assert config["_requested_state"] == "latest"
    assert handler.exposure_type == WFI_TYPE_IMAGE
    assert handler.sync_mappings is True


def test_config_context_and_exposure_type(config):
    config["crds_context"] = "roman_0042.pmap"
    config["exposure_type"] = WFI_TYPE_GRISM
    handler = CrdsHandler()
    assert handler.crds_context == "roman_0042.pmap"
    assert handler.exposure_type == WFI_TYPE_GRISM


def test_arguments_override_config(config):
    config["crds_context"] = "roman_0042.pmap"
    handler = CrdsHandler(crds_context="roman_0050.pmap", exposure_type=WFI_TYPE_GRISM)
    assert handler.crds_context == "roman_0050.pmap"
    assert handler.exposure_type == WFI_TYPE_GRISM


def test_context_state_from_config(config):
    config["context_state"] = "operational"
    CrdsHandler()
    assert config["_requested_state"] == "operational"


def test_bad_exposure_type_argument_raises(config):
    with pytest.raises(ValueError, match="exposure_type"):
        CrdsHandler(exposure_type="WFI_BOGUS")


def test_environment_set_from_config_when_unset(config, monkeypatch):
    monkeypatch.delenv("CRDS_SERVER_URL", raising=False)
    monkeypatch.delenv("CRDS_PATH", raising=False)
    config["server_url"] = "https://roman-crds.stsci.edu"
    config["crds_path"] = "/tmp/crds_cache"
    CrdsHandler()
    assert crds_handler_module.os.environ["CRDS_SERVER_URL"] == "https://roman-crds.stsci.edu"
    assert crds_handler_module.os.environ["CRDS_PATH"] == "/tmp/crds_cache"


def test_environment_wins_over_config(config, monkeypatch):
    monkeypatch.setenv("CRDS_SERVER_URL", "https://from-env.example")
    monkeypatch.setenv("CRDS_PATH", "/from/env")
    config["server_url"] = "https://roman-crds.stsci.edu"
    config["crds_path"] = "/tmp/crds_cache"
    CrdsHandler()
    assert crds_handler_module.os.environ["CRDS_SERVER_URL"] == "https://from-env.example"
    assert crds_handler_module.os.environ["CRDS_PATH"] == "/from/env"


def test_sync_skipped_when_disabled(config, monkeypatch):
    config["sync_mappings"] = False

    def fail_run(*args, **kwargs):
        raise AssertionError("crds sync should not run when sync_mappings is false")

    monkeypatch.setattr(crds_handler_module.subprocess, "run", fail_run)
    CrdsHandler(crds_context="roman_0077.pmap")._sync_context()
    assert "roman_0077.pmap" not in CrdsHandler._synced_contexts
