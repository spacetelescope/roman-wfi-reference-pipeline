from pathlib import Path

import pytest

from wfi_reference_pipeline.constants import REF_TYPE_READNOISE as REF_TYPE
from wfi_reference_pipeline.pipelines.readnoise_pipeline import ReadnoisePipeline as PipelineClass

PIPELINE_MODULE = "wfi_reference_pipeline.pipelines.readnoise_pipeline"
BASE_MODULE = "wfi_reference_pipeline.pipelines.pipeline"
STUB_CONFIG = {
    "ingest_dir": "/stub/ingest",
    "prep_dir": "/stub/prep",
    "crds_ready_dir": "/stub/crds_ready",
}
STUB_DB_CONFIG = {"use_rtbdb": False}



# Pipeline instance with all base-class I/O mocked. Portable as-is.
@pytest.fixture
def pipeline(mocker):
    mocker.patch(f"{BASE_MODULE}.configure_logging")
    mocker.patch(f"{BASE_MODULE}.get_data_files_config", return_value=STUB_CONFIG)
    mocker.patch(f"{BASE_MODULE}.get_db_config", return_value=STUB_DB_CONFIG)
    mocker.patch(f"{BASE_MODULE}.FileHandler")
    mocker.patch(f"{BASE_MODULE}.DBHandler")
    return PipelineClass("WFI01")


# Mock rdm.open + every romancal Step.call that prep_pipeline invokes.
#
# PIPELINE-SPECIFIC: mirror the `from romancal.* import *Step` block at the
# top of your pipeline module. Any unmocked step will try to run on the
# fake input and fail.
@pytest.fixture
def mock_prep_internals(mocker):
    fake_result = mocker.MagicMock()
    fake_result.meta.filename = "sample_uncal.asdf"
    mocker.patch(f"{PIPELINE_MODULE}.rdm.open")
    mocker.patch(f"{PIPELINE_MODULE}.DQInitStep.call", return_value=fake_result)
    mocker.patch(f"{PIPELINE_MODULE}.SaturationStep.call", return_value=fake_result)
    mocker.patch(f"{PIPELINE_MODULE}.LinearityStep.call", return_value=fake_result)


# Mock MakeDevMeta + the reference-type class used inside run_pipeline.
# Returns the patched reference-type class so tests can inspect call args.
#
# PIPELINE-SPECIFIC:
#   - Replace `ReadNoise` with your reference-type class (Flat / Dark /
#     Mask / ReferencePixel).
#   - Replace `meta_readnoise` with the matching MakeDevMeta attribute
#     (meta_flat / meta_dark / meta_mask / meta_referencepixel).
@pytest.fixture
def mock_run_internals(pipeline, mocker):
    meta = mocker.MagicMock()
    meta.meta_readnoise.mode = "WIM"
    meta.meta_readnoise.instrument_detector = "WFI01"
    mocker.patch(f"{PIPELINE_MODULE}.MakeDevMeta", return_value=meta)
    pipeline.file_handler.format_pipeline_output_file_path.return_value = (
        Path("/stub/out.asdf")
    )
    return mocker.patch(f"{PIPELINE_MODULE}.ReadNoise")


# ---- __init__ -------------------------------------------------------------
# Portable across pipelines as-is.

def test_init_sets_correct_ref_type(pipeline):
    assert pipeline.ref_type == REF_TYPE


def test_init_normalises_detector_to_uppercase(pipeline):
    assert PipelineClass("wfi05").detector == "WFI05"


def test_init_rejects_invalid_detector(pipeline):
    with pytest.raises(KeyError):
        PipelineClass("WFI99")


# ---- select_uncal_files ---------------------------------------------------
# Portable across pipelines as-is.

def test_select_uncal_files_populates_uncal_files(pipeline, mocker):
    fake = [Path("/stub/ingest/a.asdf"), Path("/stub/ingest/b.asdf")]
    mocker.patch.object(pipeline, "ingest_path", mocker.MagicMock())
    pipeline.ingest_path.glob.return_value = iter(fake)

    pipeline.select_uncal_files()

    assert len(pipeline.uncal_files) == len(fake)


def test_select_uncal_files_clears_stale_entries(pipeline, mocker):
    pipeline.uncal_files = ["/stub/ingest/stale.asdf"]
    mocker.patch.object(pipeline, "ingest_path", mocker.MagicMock())
    pipeline.ingest_path.glob.return_value = iter([])

    pipeline.select_uncal_files()

    assert pipeline.uncal_files == []


# ---- prep_pipeline --------------------------------------------------------
# Portable as-is once mock_prep_internals lists the right Steps.

def test_prep_pipeline_produces_one_prepped_file_per_input(pipeline, mock_prep_internals):
    pipeline.file_handler.format_prep_output_file_path.side_effect = (
        lambda name: Path(f"/stub/prep/{name}")
    )

    pipeline.prep_pipeline(file_list=["/a.asdf", "/b.asdf", "/c.asdf"])

    assert len(pipeline.prepped_files) == 3


def test_prep_pipeline_defaults_to_self_uncal_files(pipeline, mock_prep_internals):
    pipeline.uncal_files = ["/stub/ingest/from_self.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = (
        Path("/stub/prep/x.asdf")
    )

    pipeline.prep_pipeline()

    assert len(pipeline.prepped_files) == 1


def test_prep_pipeline_clears_stale_state(pipeline, mock_prep_internals):
    pipeline.prepped_files = ["/stub/prep/leftover.asdf"]
    pipeline.file_handler.format_prep_output_file_path.return_value = (
        Path("/stub/prep/new.asdf")
    )

    pipeline.prep_pipeline(file_list=["/stub/ingest/new.asdf"])

    pipeline.file_handler.remove_existing_prepped_files_for_ref_type.assert_called_once()
    assert pipeline.prepped_files == [Path("/stub/prep/new.asdf")]


# ---- run_pipeline ---------------------------------------------------------
# Defaulting + override tests are portable as-is. The outfile test names
# `make_readnoise_image`, which is PIPELINE-SPECIFIC.

def test_run_pipeline_defaults_to_prepped_files(pipeline, mock_run_internals):
    pipeline.prepped_files = [Path("/stub/prep/a.asdf")]

    pipeline.run_pipeline()

    assert mock_run_internals.call_args.kwargs["file_list"] == pipeline.prepped_files


def test_run_pipeline_uses_explicit_file_list(pipeline, mock_run_internals):
    pipeline.prepped_files = [Path("/stub/prep/do_not_use.asdf")]

    pipeline.run_pipeline(file_list=["/stub/elsewhere/use_me.asdf"])

    assert mock_run_internals.call_args.kwargs["file_list"] == [
        Path("/stub/elsewhere/use_me.asdf")
    ]


# PIPELINE-SPECIFIC: replace `make_readnoise_image` below with the
# reference-type class's image-building method
# (make_flat_from_files / make_rate_image_from_data_cube /
# make_referencepixel_image).
def test_run_pipeline_writes_outfile(pipeline, mock_run_internals):
    pipeline.prepped_files = [Path("/stub/prep/a.asdf")]

    pipeline.run_pipeline()

    instance = mock_run_internals.return_value
    instance.make_readnoise_image.assert_called_once()
    instance.generate_outfile.assert_called_once()
