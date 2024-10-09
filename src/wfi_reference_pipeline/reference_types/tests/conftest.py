import os
import pytest


@pytest.fixture(scope='session', autouse=True)
def cleanup_superdark_file():
    yield  # Tests will run
    # Cleanup step to remove the superdark file
    file_path = "WFI01_superdark.asdf"
    if os.path.isfile(file_path):
        os.remove(file_path)
