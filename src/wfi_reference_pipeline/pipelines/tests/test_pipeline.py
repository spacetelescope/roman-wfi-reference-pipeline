from pathlib import Path

import pytest

from wfi_reference_pipeline.constants import REF_TYPE_READNOISE, WFI_DETECTORS
from wfi_reference_pipeline.pipelines.pipeline import Pipeline


# Minimal concrete subclass
class MinimalPipeline(Pipeline):
    def select_uncal_files(self): pass
    def prep_pipeline(self, file_list=None): pass
    def run_pipeline(self, file_list=None): pass
    def pre_deliver(self): pass
    def deliver(self): pass

# Fake 
STUB_CONFIG = {
    "ingest_dir": "/stub/ingest",
    "prep_dir": "/stub/prep",
    "crds_ready_dir": "/stub/crds_ready",
}

STUB_DB_CONFIG = {
    "use_rtbdb": False,
    "use_dsn": False,
    "sql_server_str": None,
    "sql_database_str": None,
    "port": None,
    "dsn_header_str": None,
}

BASE = "wfi_reference_pipeline.pipelines.pipeline"

# Putting all necessary mocks within one fixture
@pytest.fixture
def pipeline_patches(mocker):
    """Patch all external dependencies so Pipeline.__init__ runs without I/O."""
    mocker.patch(f"{BASE}.configure_logging")
    mocker.patch(f"{BASE}.get_data_files_config", return_value=STUB_CONFIG)
    mocker.patch(f"{BASE}.get_db_config", return_value=STUB_DB_CONFIG)
    mocker.patch(f"{BASE}.FileHandler")
    mocker.patch(f"{BASE}.DBHandler")

# A fully initialised MinimalPipeline using WFI01 and REF_TYPE_READNOISE
@pytest.fixture
def pipeline(pipeline_patches):
    return MinimalPipeline(REF_TYPE_READNOISE, "WFI01")



def test_cannot_instantiate_pipeline_directly_passes(pipeline_patches):
    with pytest.raises(TypeError):
        Pipeline(REF_TYPE_READNOISE, "WFI01")


### Tests on Proper Attributes Set ###


@pytest.mark.parametrize("detector", sorted(WFI_DETECTORS))
def test_init_all_valid_detectors_accepted_passes(detector, pipeline_patches):
    """Every detector in WFI_DETECTORS should initialise without error."""
    p = MinimalPipeline(REF_TYPE_READNOISE, detector)
    assert p.detector == detector.upper()


def test_init_mixed_case_detector_normalised_passes(pipeline_patches):
    p = MinimalPipeline(REF_TYPE_READNOISE, "Wfi05")
    assert p.detector == "WFI05"


def test_init_invalid_detector_raises_key_error_passes(pipeline_patches):
    with pytest.raises(KeyError):
        MinimalPipeline(REF_TYPE_READNOISE, "WFI99")


def test_init_empty_detector_raises_key_error_passes(pipeline_patches):
    with pytest.raises(KeyError):
        MinimalPipeline(REF_TYPE_READNOISE, "")


### Tests on Proper Attributes Set ###


def test_init_ref_type_stored_passes(pipeline):
    assert pipeline.ref_type == REF_TYPE_READNOISE


def test_init_uncal_files_is_empty_list_passes(pipeline):
    assert pipeline.uncal_files == []


def test_init_prepped_files_is_empty_list_passes(pipeline):
    assert pipeline.prepped_files == []


def test_init_ingest_path_from_config_passes(pipeline):
    assert pipeline.ingest_path == Path(STUB_CONFIG["ingest_dir"])


def test_init_prep_path_from_config_passes(pipeline):
    assert pipeline.prep_path == Path(STUB_CONFIG["prep_dir"])


def test_init_pipeline_out_path_from_config_passes(pipeline):
    assert pipeline.pipeline_out_path == Path(STUB_CONFIG["crds_ready_dir"])


### Config Failure Tests ###


def test_init_bad_config_calls_sys_exit_passes(mocker):
    """A missing or invalid config file causes __init__ to call sys.exit."""
    mocker.patch(f"{BASE}.configure_logging")
    mocker.patch(f"{BASE}.get_data_files_config", side_effect=FileNotFoundError)
    mocker.patch(f"{BASE}.get_db_config", return_value=STUB_DB_CONFIG)

    with pytest.raises(SystemExit):
        MinimalPipeline(REF_TYPE_READNOISE, "WFI01")



### Potentially FileHandler/DBHandler initialisation. These would have to be more whitebox tests ###


### Tests to fail init_quality_control since it is not implemented yet

@pytest.mark.xfail(raises=ValueError, reason="init_quality_control is not implemented for REF_TYPE_READNOISE")
def test_init_quality_control_works_for_readnoise_fails(pipeline):
    pipeline.init_quality_control()

