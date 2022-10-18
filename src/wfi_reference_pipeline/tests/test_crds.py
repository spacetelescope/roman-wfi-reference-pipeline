import pytest
from ..utilities import submit_files


def test_exceptions():
    """
    Test that we get the expected exceptions for bad inputs.
    """

    si = {'description': 'test'}

    # Cannot find input files.
    with pytest.raises(FileNotFoundError):
        _ = submit_files.WFIsubmit(['bad_file_input.asdf'],
                                     submission_info=si)

    # Wrong type of CRDS server.
    with pytest.raises(ValueError):
        _ = submit_files.WFIsubmit([], submission_info=si,
                                     server='bad')

    # Input files not given as list.
    with pytest.raises(TypeError):
        _ = submit_files.WFIsubmit('bad_file_input.asdf',
                                     submission_info=si)

    # Form details not supplied.
    with pytest.raises(TypeError):
        _ = submit_files.WFIsubmit([])

    # Empty input file list.
    with pytest.raises(ValueError):
        _ = submit_files.WFIsubmit([], submission_info=si)
