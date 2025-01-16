from subprocess import CalledProcessError
from unittest.mock import patch, Mock

import pytest

TESTED_MODULE = "src.api.main"


@patch("src.api.security.SecretManagerServiceClient", Mock())
@patch("src.api.security.retrieve_secret", Mock(return_value="valid-token"))
@patch(f"{TESTED_MODULE}.gc.collect")
@patch(f"{TESTED_MODULE}.shutil.rmtree")
@patch(f"{TESTED_MODULE}.flair_detect_service")
@pytest.mark.parametrize(
    "model, token, result_flair_detect_service, clean_all_files, "
    "expected_status_code, expected_gc_collect_count, "
    "expected_shutil_rmtree_count",
    [
        # Case 1 : ok without cleaning files
        ("rgbi-15cl-resnet34-unet", "valid-token", {}, False, 200, 1, 0),
        # Case 2 : ok
        ("rgbi-15cl-resnet34-unet", "valid-token", {}, True, 200, 1, 2),
        # Case 3 : model unknown
        ("unknown model", "valid-token", {}, True, 422, 0, 0),
        # Case 4 : unvalid token
        ("rgbi-15cl-resnet34-unet", "unvalid-token", {}, True, 401, 0, 0),
        # Case 5 : Unknown exeption during flair detect service
        ("rgbi-15cl-resnet34-unet", "valid-token", Exception, True, 500, 1, 2),
        # Case 6 : CalledProcessError exception during flair detect service
        (
            "rgbi-15cl-resnet34-unet",
            "valid-token",
            CalledProcessError,
            True,
            500,
            1,
            2,
        ),
    ],
)
def test_flair_detect_app(
    flair_detect_service_mock,
    shutil_rmtree_mock,
    gc_collect_mock,
    model,
    token,
    result_flair_detect_service,
    clean_all_files,
    expected_status_code,
    expected_gc_collect_count,
    expected_shutil_rmtree_count,
    test_client,
):
    # init
    params = {
        "image_bucket_name": "netcarbon-ortho",
        "image_blob_path": "RGBN/tile_RGBN.tif",
        "model": model,
        "output_bucket_name": "netcarbon-landcover",
        "output_blob_path": "prediction_raster/tile_flair-model-test.tif",
        "prediction_id": "RGBN-crc32c_flair-model-test",
    }

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    # mock result for flair detect service
    if isinstance(result_flair_detect_service, dict):
        flair_detect_service_mock.return_value = result_flair_detect_service
    else:
        flair_detect_service_mock.side_effect = result_flair_detect_service

    # act
    with patch(
        f"{TESTED_MODULE}.CLEAN_ALL_FILES_AFTER_PREDICTION", clean_all_files
    ):
        response = test_client.post(
            "/flair-detect", headers=headers, params=params
        )

    # assert
    assert response.status_code == expected_status_code
    assert gc_collect_mock.call_count == expected_gc_collect_count
    assert shutil_rmtree_mock.call_count == expected_shutil_rmtree_count
