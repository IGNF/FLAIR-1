import os
from unittest.mock import Mock, patch

import pytest

from src.api.flair_detect_service import get_requested_model, get_output_prediction_folder, download_file_to_process
from tests.tests_constants import TESTS_DATA_FOLDER


TESTED_MODULE = "src.api.flair_detect_service"

FLAIR_MODEL_MOCK = Mock(
    relative_weights_path="flair-model-test_weights.pth",
    bucket_name="bucket-test",
    blob_prefix="blob-test",
)

TEST_AVAILABLE_MODELS = {"flair-model-test": FLAIR_MODEL_MOCK}


@patch(f"{TESTED_MODULE}.OUTPUT_FOLDER", "/output/folder")
def test_get_output_prediction_folder():
    prediction_id = "RGBN-crc32c_model-name"

    folder = get_output_prediction_folder(prediction_id)

    assert folder == f"/output/folder/{prediction_id}"


@pytest.mark.parametrize(
    "existing_paths, "
    "supported_model, "
    "data_folder, "
    "expected_download_count, "
    "expected_flair_model, "
    "expected_model_weights_path",
    [
        # Case 1 : model already exists locally
        (
            [os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth")],
            "flair-model-test",
            TESTS_DATA_FOLDER,
            0,
            FLAIR_MODEL_MOCK,
            os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth"),
        ),
        # Case 2 : model doesn't exist locally
        (
            [],
            "flair-model-test",
            TESTS_DATA_FOLDER,
            1,
            FLAIR_MODEL_MOCK,
            os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth"),
        ),
    ],
)
def test_get_requested_model(
    existing_paths,
    supported_model,
    data_folder,
    expected_download_count,
    expected_flair_model,
    expected_model_weights_path,
):
    # init
    client_gcs_mock = Mock()

    # mock os.path.exists
    os_path_exists_mock = Mock()

    def fake_os_path_exists(path: str):
        return path in existing_paths

    os_path_exists_mock.side_effect = fake_os_path_exists

    # act
    with patch(f"{TESTED_MODULE}.available_models", TEST_AVAILABLE_MODELS):
        with patch(f"{TESTED_MODULE}.os.path.exists", os_path_exists_mock):
            with patch(
                f"{TESTED_MODULE}.download_gcs_folder"
            ) as download_gcs_folder_mock:
                flair_model, model_weights_path = get_requested_model(
                    model=supported_model,
                    client=client_gcs_mock,
                    data_folder=data_folder,
                )

    # assert
    assert download_gcs_folder_mock.call_count == expected_download_count
    assert flair_model == expected_flair_model
    assert model_weights_path == expected_model_weights_path


@patch(f"{TESTED_MODULE}.download_file")
def test_download_file_to_process(
    download_file_mock
):
    # init
    client_gcs = Mock()

    # act
    image_local_path = download_file_to_process(
        image_bucket_name="netcarbon-ortho",
        image_blob_path="RGBN/tile_RGBN.tif",
        client=client_gcs,
        input_folder="/data/input"
    )

    # assert
    assert image_local_path == "/data/input/tile_RGBN.tif"
    download_file_mock.assert_called_once()
