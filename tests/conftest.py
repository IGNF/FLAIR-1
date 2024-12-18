import os
from shutil import rmtree

import pytest

from tests.tests_constants import TESTS_DATA_FOLDER


@pytest.fixture
def tests_output_folder():
    # Create output folder for tests
    output_folder = os.path.join(TESTS_DATA_FOLDER,
                                 "output")
    os.makedirs(output_folder, exist_ok=True)

    # Return path to the folder
    yield output_folder

    # Remove folder after tests
    rmtree(output_folder, ignore_errors=True)
