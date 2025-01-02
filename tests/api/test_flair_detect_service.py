import os
from unittest.mock import Mock, patch

import pytest

from src.api.flair_detect_service import (
    get_requested_model,
    get_output_prediction_folder,
    download_file_to_process,
    upload_result_to_bucket,
    flair_detect_service,
)
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
    "expected_model_weights_path",
    [
        # Case 1 : model already exists locally
        (
            [os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth")],
            "flair-model-test",
            TESTS_DATA_FOLDER,
            0,
            os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth"),
        ),
        # Case 2 : model doesn't exist locally
        (
            [],
            "flair-model-test",
            TESTS_DATA_FOLDER,
            1,
            os.path.join(TESTS_DATA_FOLDER, "flair-model-test_weights.pth"),
        ),
    ],
)
def test_get_requested_model(
    existing_paths,
    supported_model,
    data_folder,
    expected_download_count,
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
                model_weights_path = get_requested_model(
                    model=supported_model,
                    client=client_gcs_mock,
                    data_folder=data_folder,
                )

    # assert
    assert download_gcs_folder_mock.call_count == expected_download_count
    assert model_weights_path == expected_model_weights_path


@patch(f"{TESTED_MODULE}.download_file")
def test_download_file_to_process(download_file_mock):
    # init
    client_gcs = Mock()

    # act
    image_local_path = download_file_to_process(
        image_bucket_name="netcarbon-ortho",
        image_blob_path="RGBN/tile_RGBN.tif",
        client=client_gcs,
        input_folder="/data/input",
    )

    # assert
    assert image_local_path == "/data/input/tile_RGBN.tif"
    download_file_mock.assert_called_once()


@patch(f"{TESTED_MODULE}.upload_file")
def test_upload_result_to_bucket(upload_file_mock):
    # init
    output_prediction_folder = "/data/output/prediction"
    output_name = "tile_model.tif"
    output_bucket_name = "netcarbon-landcover"
    output_blob_path = "prediction_raster"
    client_gcs = Mock()

    # act
    upload_result_to_bucket(
        output_prediction_folder=output_prediction_folder,
        output_name=output_name,
        output_bucket_name=output_bucket_name,
        output_blob_path=output_blob_path,
        client=client_gcs,
    )

    # assert
    upload_file_mock.assert_called_once_with(
        bucket_name=output_bucket_name,
        blob_path=output_blob_path,
        local_path=f"{output_prediction_folder}/{output_name}",
        client=client_gcs,
    )


@patch(f"{TESTED_MODULE}.FLAIR_DETECT_BATCH_SIZE", 4)
@patch(f"{TESTED_MODULE}.upload_result_to_bucket")
@patch(f"{TESTED_MODULE}.run_prediction")
@patch(f"{TESTED_MODULE}.perf_counter")
@patch(f"{TESTED_MODULE}.torch")
@patch(f"{TESTED_MODULE}.setup_config_flair_detect")
@patch(f"{TESTED_MODULE}.download_file_to_process")
@patch(f"{TESTED_MODULE}.get_requested_model")
@patch(f"{TESTED_MODULE}.Client")
@patch(f"{TESTED_MODULE}.os.makedirs")
@patch(f"{TESTED_MODULE}.get_output_prediction_folder")
def test_flair_detect_service(
    get_output_prediction_folder_mock,
    os_makedirs_mock,
    client_mock,
    get_requested_model_mock,
    download_file_to_process_mock,
    setup_config_flair_detect_mock,
    torch_mock,
    perf_counter_mock,
    run_prediction_mock,
    upload_result_to_bucket_mock,
):
    # init
    image_bucket_name_test = "netcarbon-ortho"
    image_blob_path_test = "RGBN/tile_RGBN.tif"
    model_test = "flair-model-test"
    output_bucket_name_test = "netcarbon-landcover"
    output_blob_path_test = "prediction_raster/tile_flair-model-test.tif"
    prediction_id_test = "RGBN-crc32c_flair-model-test"

    # mock output prediction folder
    output_prediction_folder_mock = Mock()
    get_output_prediction_folder_mock.return_value = (
        output_prediction_folder_mock
    )

    # mock GCS client
    client_gcs_mock = Mock()
    client_mock.return_value = client_gcs_mock

    # mock model
    model_weights_path_mock = Mock()
    get_requested_model_mock.return_value = model_weights_path_mock

    # mock image local path
    image_local_path_mock = Mock()
    download_file_to_process_mock.return_value = image_local_path_mock

    # mock flair-detect config
    prediction_config_path_mock = Mock()
    setup_config_flair_detect_mock.return_value = prediction_config_path_mock

    # act
    response = flair_detect_service(
        image_bucket_name=image_bucket_name_test,
        image_blob_path=image_blob_path_test,
        model=model_test,
        output_bucket_name=output_bucket_name_test,
        output_blob_path=output_blob_path_test,
        prediction_id=prediction_id_test,
    )

    # assert
    get_output_prediction_folder_mock.assert_called_once_with(
        prediction_id=prediction_id_test
    )
    get_requested_model_mock.assert_called_once_with(
        model=model_test, client=client_gcs_mock
    )
    download_file_to_process_mock.assert_called_once_with(
        image_bucket_name=image_bucket_name_test,
        image_blob_path=image_blob_path_test,
        client=client_gcs_mock,
    )
    setup_config_flair_detect_mock.assert_called_once_with(
        input_image_path=image_local_path_mock,
        model_weights_path=model_weights_path_mock,
        output_image_name="tile_flair-model-test.tif",
        output_folder=output_prediction_folder_mock,
        batch_size=4,
    )
    run_prediction_mock.assert_called_once_with(
        prediction_config_path=prediction_config_path_mock
    )
    upload_result_to_bucket_mock.assert_called_once_with(
        output_prediction_folder=output_prediction_folder_mock,
        output_name="tile_flair-model-test.tif",
        output_bucket_name=output_bucket_name_test,
        output_blob_path=output_blob_path_test,
        client=client_gcs_mock,
    )
    assert response["prediction_id"] == prediction_id_test
    assert (
        f"{output_bucket_name_test}/{output_blob_path_test}"
        in response["message"]
    )
