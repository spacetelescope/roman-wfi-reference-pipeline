from pathlib import Path

import pytest

from wfi_reference_pipeline.constants import REF_TYPE_READNOISE
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline

# Constants to help with mocking
PIPELINE_MODULE = "wfi_reference_pipeline.pipelines.readnoise_pipeline"
BASE_MODULE = "wfi_reference_pipeline.pipelines.pipeline"
STUB_CONFIG = {
    "ingest_dir": "/stub/ingest",
    "prep_dir": "/stub/prep",
    "crds_ready_dir": "/stub/crds_ready",
}
STUB_DB_CONFIG = {"use_rtbdb": False}


@pytest.fixture
def pipeline(mocker):
    """
    Pipeline instance with all base-class I/O mocked
    """
    mocker.patch(f"{BASE_MODULE}.configure_logging")
    mocker.patch(f"{BASE_MODULE}.get_data_files_config", return_value=STUB_CONFIG)
    mocker.patch(f"{BASE_MODULE}.get_db_config", return_value=STUB_DB_CONFIG)
    mocker.patch(f"{BASE_MODULE}.FileHandler")
    mocker.patch(f"{BASE_MODULE}.DBHandler")
    return ReadnoisePipeline("WFI01")


@pytest.fixture
def mock_prep_internals(mocker):
    """
    Mock rdm.open and every romancal Step.call that prep_pipeline invokes
    Patches take effect for any test that requests this fixture, even if the returned object isn't referenced
    """
    fake_result = mocker.MagicMock()
    fake_result.meta.filename = "sample_uncal.asdf"
    mocker.patch(f"{PIPELINE_MODULE}.rdm.open")
    mocker.patch(f"{PIPELINE_MODULE}.DQInitStep.call", return_value=fake_result)
    mocker.patch(f"{PIPELINE_MODULE}.SaturationStep.call", return_value=fake_result)
    mocker.patch(f"{PIPELINE_MODULE}.LinearityStep.call", return_value=fake_result)


### __init__ tests ###

def test_init_sets_correct_ref_type_pass(pipeline):
    """
    Pipeline should report the correct readnoise reference type
    """
    assert pipeline.ref_type == REF_TYPE_READNOISE


def test_init_normalises_detector_to_uppercase_pass(pipeline):
    """
    Detector IDs should be stored in canonical uppercase
    """
    assert ReadnoisePipeline("wfi05").detector == "WFI05"


def test_init_rejects_invalid_detector_pass(pipeline):
    """
    Unknown detector IDs should be rejected
    """
    with pytest.raises(KeyError):
        ReadnoisePipeline("WFI99")


### select_uncal_files tests ###
# TODO: More tests need to be added as the pipeline is developed about the files in uncal_files


def test_select_uncal_files_sets_uncal_files_not_none_pass(pipeline, mocker):
    """
    After selection runs, uncal_files should be set even if no matching files were found
    """
    pipeline.ingest_path = mocker.MagicMock()
    pipeline.ingest_path.glob.return_value = iter([])

    pipeline.select_uncal_files()

    assert pipeline.uncal_files is not None


### prep_pipeline tests ###

def test_prep_pipeline_produces_one_prepped_file_per_input_pass(pipeline, mock_prep_internals):
    """
    Prep should produce exactly one prepped output per input file
    """
    pipeline.file_handler.format_prep_output_file_path.side_effect = (
        lambda name: Path(f"/stub/prep/{name}")
    )

    pipeline.prep_pipeline(file_list=["/a.asdf", "/b.asdf", "/c.asdf"])

    assert len(pipeline.prepped_files) == 3


def test_prep_pipeline_defaults_to_self_uncal_files_pass(pipeline, mock_prep_internals):
    """
    When no file list is passed in, prep should fall back to self.uncal_files
    """
    pipeline.uncal_files = ["/stub/ingest/from_self.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = (
        Path("/stub/prep/x.asdf")
    )

    pipeline.prep_pipeline()

    assert len(pipeline.prepped_files) == 1


def test_prep_pipeline_clears_stale_state_pass(pipeline, mock_prep_internals):
    """
    Re-running prep should clear leftover prepped files from a prior run before writing new ones
    """
    pipeline.prepped_files = ["/stub/prep/leftover.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = (
        Path("/stub/prep/new.asdf")
    )

    pipeline.prep_pipeline(file_list=["/stub/ingest/new.asdf"])

    pipeline.file_handler.remove_existing_prepped_files_for_ref_type.assert_called_once()
    assert pipeline.prepped_files == [Path("/stub/prep/new.asdf")]


### run_pipeline tests ###
# TODO: More tests need to be added as the pipeline is developed