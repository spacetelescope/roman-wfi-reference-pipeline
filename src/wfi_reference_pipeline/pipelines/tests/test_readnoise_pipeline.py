from pathlib import Path

import pytest

from wfi_reference_pipeline.constants import REF_TYPE_READNOISE
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline

# Stub configs so Pipeline.__init__ doesn't try to read real YAML off disk.
STUB_CONFIG = {
    "ingest_dir": "/stub/ingest",
    "prep_dir": "/stub/prep",
    "crds_ready_dir": "/stub/crds_ready",
}
STUB_DB_CONFIG = {"use_rtbdb": False}

BASE = "wfi_reference_pipeline.pipelines.pipeline"
RN = "wfi_reference_pipeline.pipelines.readnoise_pipeline"


### Fixtures ###

@pytest.fixture
def pipeline(mocker):
    """A fully-initialized ReadnoisePipeline with all base-class I/O mocked away."""
    mocker.patch(f"{BASE}.configure_logging")
    mocker.patch(f"{BASE}.get_data_files_config", return_value=STUB_CONFIG)
    mocker.patch(f"{BASE}.get_db_config", return_value=STUB_DB_CONFIG)
    mocker.patch(f"{BASE}.FileHandler")
    mocker.patch(f"{BASE}.DBHandler")
    return ReadnoisePipeline("WFI01")


@pytest.fixture
def stub_glob(pipeline, mocker):
    """Returns a setter: call stub_glob([Path(...), ...]) to control glob results."""
    mocker.patch.object(pipeline, "ingest_path", mocker.MagicMock())
    return lambda files: pipeline.ingest_path.glob.configure_mock(return_value=iter(files))


@pytest.fixture
def prep_patches(mocker):
    """Mock the romancal Steps and rdm.open. Returns the three Step.call mocks."""
    mocker.patch(f"{RN}.rdm.open")
    fake_result = mocker.MagicMock()
    fake_result.meta.filename = "sample_uncal.asdf"
    return {
        "dq_init": mocker.patch(f"{RN}.DQInitStep.call", return_value=fake_result),
        "saturation": mocker.patch(f"{RN}.SaturationStep.call", return_value=fake_result),
        "linearity": mocker.patch(f"{RN}.LinearityStep.call", return_value=fake_result),
    }


@pytest.fixture
def run_patches(pipeline, mocker):
    """Mock ReadNoise + MakeDevMeta + outfile path. Returns the ReadNoise class mock."""
    meta = mocker.MagicMock()
    meta.meta_readnoise.mode = "WIM"
    meta.meta_readnoise.instrument_detector = "WFI01"
    mocker.patch(f"{RN}.MakeDevMeta", return_value=meta)
    pipeline.file_handler.format_pipeline_output_file_path.return_value = Path("/stub/out.asdf")
    return mocker.patch(f"{RN}.ReadNoise")


# ---- __init__ ---------------------------------------------------------------

def test_init_sets_ref_type_to_readnoise(pipeline):
    # Wrong ref_type silently delivers under the wrong CRDS reftype - critical.
    assert pipeline.ref_type == REF_TYPE_READNOISE


def test_init_normalizes_detector_to_uppercase(pipeline):
    assert ReadnoisePipeline("wfi05").detector == "WFI05"


def test_init_invalid_detector_raises(pipeline):
    # Fail loud at construction, not at some confusing downstream step.
    with pytest.raises(KeyError):
        ReadnoisePipeline("WFI99")


# ---- select_uncal_files -----------------------------------------------------

def test_select_uncal_files_stores_glob_results(pipeline, stub_glob):
    # Whatever glob finds must land on self.uncal_files - prep_pipeline reads from there.
    fake = [Path("/stub/ingest/a.asdf"), Path("/stub/ingest/b.asdf")]
    stub_glob(fake)

    pipeline.select_uncal_files()

    assert pipeline.uncal_files == [str(f) for f in fake]


def test_select_uncal_files_clears_stale_entries(pipeline, stub_glob):
    # Re-running selection should NOT append to old results.
    pipeline.uncal_files = ["/stub/ingest/stale.asdf"]
    stub_glob([Path("/stub/ingest/fresh.asdf")])

    pipeline.select_uncal_files()

    assert pipeline.uncal_files == ["/stub/ingest/fresh.asdf"]


def test_select_uncal_files_empty_glob_yields_empty_list(pipeline, stub_glob):
    # No matches -> empty list, not None, not error.
    stub_glob([])

    pipeline.select_uncal_files()

    assert pipeline.uncal_files == []


# ---- prep_pipeline ----------------------------------------------------------

def test_prep_pipeline_runs_all_three_romancal_steps(pipeline, prep_patches):
    # Order matters scientifically: DQInit -> Saturation -> Linearity.
    # Saturation flags must be set before linearity correction, or saturated
    # pixels get "corrected" into garbage.
    pipeline.prep_pipeline(file_list=["/stub/ingest/one.asdf"])

    prep_patches["dq_init"].assert_called_once()
    prep_patches["saturation"].assert_called_once()
    prep_patches["linearity"].assert_called_once()


def test_prep_pipeline_produces_one_prepped_file_per_input(pipeline, prep_patches):
    # Catches the classic bug where the append lives inside an if-branch
    # and silently drops files.
    pipeline.file_handler.format_prep_output_file_path.side_effect = (
        lambda name: Path(f"/stub/prep/{name}_PREPPED.asdf")
    )

    pipeline.prep_pipeline(file_list=["/a.asdf", "/b.asdf", "/c.asdf"])

    assert len(pipeline.prepped_files) == 3


def test_prep_pipeline_defaults_to_self_uncal_files(pipeline, prep_patches):
    # No args (e.g. from restart_pipeline) -> falls back to select_uncal_files output.
    pipeline.uncal_files = ["/stub/ingest/from_self.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = Path("/stub/prep/x_PREPPED.asdf")

    pipeline.prep_pipeline()

    assert len(pipeline.prepped_files) == 1


def test_prep_pipeline_clears_state_from_previous_runs(pipeline, prep_patches):
    # Both in-memory list AND on-disk PREPPED files must be reset.
    pipeline.prepped_files = ["/stub/prep/leftover_PREPPED.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = Path("/stub/prep/new_PREPPED.asdf")

    pipeline.prep_pipeline(file_list=["/stub/ingest/new.asdf"])

    pipeline.file_handler.remove_existing_prepped_files_for_ref_type.assert_called_once()
    assert pipeline.prepped_files == [Path("/stub/prep/new_PREPPED.asdf")]


# ---- run_pipeline -----------------------------------------------------------

def test_run_pipeline_defaults_to_prepped_files(pipeline, run_patches):
    # No file_list given -> use self.prepped_files (the restart_pipeline path).
    pipeline.prepped_files = [Path("/stub/prep/a_PREPPED.asdf")]

    pipeline.run_pipeline()

    assert run_patches.call_args.kwargs["file_list"] == pipeline.prepped_files


def test_run_pipeline_respects_explicit_file_list(pipeline, run_patches):
    # Explicit list overrides self.prepped_files so users can re-run just the final stage.
    pipeline.prepped_files = [Path("/stub/prep/do_NOT_use.asdf")]

    pipeline.run_pipeline(file_list=["/stub/elsewhere/use_me.asdf"])

    assert run_patches.call_args.kwargs["file_list"] == [Path("/stub/elsewhere/use_me.asdf")]


def test_run_pipeline_builds_image_and_writes_outfile(pipeline, run_patches):
    # Skipping either call means the pipeline "succeeds" but no usable file is produced.
    pipeline.prepped_files = [Path("/stub/prep/a_PREPPED.asdf")]

    pipeline.run_pipeline()

    instance = run_patches.return_value
    instance.make_readnoise_image.assert_called_once()
    instance.generate_outfile.assert_called_once()
