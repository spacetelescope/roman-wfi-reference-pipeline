import pytest
import tempfile
import os
from wfi_reference_pipeline.utilities.submit_files_to_crds import WFISubmit


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


def test_with_temp_file(temp_file):
    """
    Test WFISubmit with a valid temporary file.
    """
    si = {'description': 'test', 'file_type': 'MASK'}

    # Initialize WFISubmit with temp_file
    wfis = WFISubmit([temp_file], submission_info=si)

    # Assert temp_file
    assert wfis.files == [temp_file]


def test_exceptions(temp_file):
    """
    Test that we get the expected exceptions for bad inputs with a valid temporary file.
    """
    si = {'description': 'test', 'file_type': 'MASK'}

    # Testing with empty file list should still raise ValueError
    with pytest.raises(ValueError):
        _ = WFISubmit([], submission_info=si)

    # Wrong type of CRDS server.
    with pytest.raises(ValueError):
        _ = WFISubmit([temp_file], submission_info=si, server='bad')

    # Input files not given as list.
    with pytest.raises(TypeError):
        _ = WFISubmit('bad_file_input.asdf', submission_info=si)

    # Form details not supplied.
    with pytest.raises(ValueError):
        _ = WFISubmit([], submission_info=None)
