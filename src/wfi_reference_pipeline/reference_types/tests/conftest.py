import os
import subprocess

import pytest


@pytest.fixture(scope='session', autouse=True)
def cleanup_superdark_file():
    yield  # Tests will run
    # Cleanup step to remove the superdark file
    file_path = "WFI01_superdark.asdf"
    if os.path.isfile(file_path):
        os.remove(file_path)


@pytest.fixture
def run_make_pars(tmp_path, monkeypatch):
    """
    Runs make_pars_files.py in an isolated temp directory.
    Returns subprocess result and ensures cwd isolation.
    """
    monkeypatch.chdir(tmp_path)

    def _run():
        return subprocess.run(
            ["python", "make_pars_files.py"],
            capture_output=True,
            text=True,
        )

    return _run, tmp_path
