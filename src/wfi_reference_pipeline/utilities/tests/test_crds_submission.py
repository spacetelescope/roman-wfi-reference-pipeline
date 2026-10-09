import os
import tempfile

import pytest

from wfi_reference_pipeline.utilities.submit_files_to_crds import WFISubmit, SubmissionForm

skip_on_github = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip this test on GitHub Actions, no crds access"
)

full_config = {
    "submission_form": {
        "deliverer": "test deliverer",
        "other_email": "test other_email",
        "instrument": "test instrument",
        "file_type": "test file_type",
        "history_updated": False,
        "pedigree_updated": False,
        "keywords_checked": False,
        "descrip_updated": False,
        "useafter_updated": False,
        "useafter_matches": "test useafter_matches",
        "compliance_verified": "test compliance_verified",
        "etc_delivery": False,
        "calpipe_version": "test calpipe_version",
        "replacement_files": False,
        "old_reference_files": ["test reference_file1", "test reference_file2"],
        "replacing_badfiles": "test replacing_badfiles",
        "jira_issue": ["test jira_issue1", "test jira_issue2"],
        "table_rows_changed": "test table_rows_changed",
        "reprocess_affected": False,
        "modes_affected": "test modes_affected",
        "change_level": "test change_level",
        "correctness_testing": "test correctness_testing",
        "additional_considerations": "test additional_considerations",
        "description": "test description",
    }
}
partial_config = {
    "submission_form": {
        "deliverer": "test deliverer",
        "other_email": "test other_email",
        "instrument": "test instrument",
        "file_type": "test file_type",
        "replacement_files": False,
        "old_reference_files": ["test reference_file1", "test reference_file2"],
    }
}


@pytest.fixture
def temp_file():
    """
    Create a temporary file for testing.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b'Test content')  # You can write any content needed for your test
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    os.remove(tmp_file_path)  # Clean up the temporary file after the test

@skip_on_github
def test_with_temp_file(temp_file):
    """
    Test WFISubmit with a valid temporary file.
    """
    si = {'description': 'test', 'file_type': 'MASK'}

    # Initialize WFISubmit with temp_file
    wfis = WFISubmit([temp_file], form_info=si)

    # Assert temp_file
    assert wfis.files == [temp_file]


def test_exceptions(temp_file):
    """
    Test that we get the expected exceptions for bad inputs with a valid temporary file.
    """
    si = {'description': 'test', 'file_type': 'MASK'}

    # Testing with empty file list should still raise ValueError
    with pytest.raises(ValueError):
        _ = WFISubmit([], form_info=si)

    # Wrong type of CRDS server.
    with pytest.raises(ValueError):
        _ = WFISubmit([temp_file], form_info=si, server='bad')

    # Input files not given as list.
    with pytest.raises(TypeError):
        _ = WFISubmit('bad_file_input.asdf', form_info=si)

    # Form details not supplied.
    with pytest.raises(ValueError):
        _ = WFISubmit([], form_info=None)

def test_submission_form_good_config_passes(monkeypatch):

    monkeypatch.setattr("wfi_reference_pipeline.utilities.submit_files_to_crds.get_crds_submission_config", lambda: full_config)

    submission_form = SubmissionForm()

    for key, value in full_config["submission_form"].items():
        submission_form_value = getattr(submission_form, key)
        assert submission_form_value == value

def test_submission_form_incomplete_config_passes(monkeypatch):
    monkeypatch.setattr("wfi_reference_pipeline.utilities.submit_files_to_crds.get_crds_submission_config", lambda: partial_config)
    submission_form = SubmissionForm()

    assert submission_form.deliverer == partial_config["submission_form"]["deliverer"]
    # Test default value
    assert submission_form.change_level == "MODERATE"