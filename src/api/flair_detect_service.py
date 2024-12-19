import os
from typing import Tuple
from subprocess import CalledProcessError, run

from fastapi import HTTPException
from google.cloud.exceptions import NotFound, Forbidden
from google.cloud.storage import Client

from src.api.handle_files import (
    download_gcs_folder,
    download_file,
    upload_file,
)
from src.api.logger import get_logger
from src.api.classes.prediction_models import (
    SupportedModel,
    Rgbi15clResnet34UnetModel,
    FlairModel,
)
from src.constants import DATA_FOLDER, INPUT_FOLDER

logger = get_logger()

# Models hash map
available_models: dict[SupportedModel, FlairModel] = {
    SupportedModel.Rgbi15clResnet34Unet: Rgbi15clResnet34UnetModel()
}


def get_requested_model(
    model: SupportedModel, client: Client, data_folder: str = DATA_FOLDER
) -> Tuple[FlairModel, str]:
    # Get model from hash map
    flair_model = available_models[model]
    model_weights_path = os.path.join(
        data_folder, flair_model.relative_weights_path
    )

    # Download model from gcs
    if not os.path.exists(model_weights_path):
        download_gcs_folder(
            bucket_name=flair_model.bucket_name,
            blob_prefix=flair_model.blob_prefix,
            local_directory=data_folder,
            client=client,
        )
    logger.info("Flair model weights available at %s", model_weights_path)

    return flair_model, model_weights_path


def download_file_to_process(
    image_bucket_name: str,
    image_blob_path: str,
    client: Client,
    input_folder: str = INPUT_FOLDER,
):
    image_name = os.path.basename(image_blob_path)
    image_local_path = os.path.join(input_folder, image_name)
    try:
        download_file(
            bucket_name=image_bucket_name,
            blob_path=image_blob_path,
            local_path=image_local_path,
            client=client,
        )
    except NotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"Blob {image_blob_path} in bucket {image_bucket_name} \
                    not found : {str(e)}",
        ) from e
    except Forbidden as e:
        raise HTTPException(
            status_code=403,
            detail=f"Permission error during download of blob \
                    {image_blob_path} in bucket {image_bucket_name} \
                    : {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected exception during download of blob \
                {image_blob_path} in bucket {image_bucket_name} : {str(e)}",
        ) from e

    logger.info(
        "Blob %s in bucket %s downloaded at %s",
        image_blob_path,
        image_bucket_name,
        image_local_path,
    )

    return image_local_path


def run_prediction(prediction_config_path: str):
    try:
        result = run(
            ["flair-detect", "--conf", prediction_config_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing flair-detect script: {e.stderr.strip()}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected exception during flair-detect script : \
                {str(e)}",
        ) from e

    logger.info(
        "Prediction with config %s done with success", prediction_config_path
    )

    return result


def upload_result_to_bucket(
    output_prediction_folder: str,
    output_name: str,
    output_bucket_name: str,
    output_blob_path: str,
    client: Client,
):
    output_path = os.path.join(output_prediction_folder, output_name)
    try:
        upload_file(
            bucket_name=output_bucket_name,
            blob_path=output_blob_path,
            local_path=output_path,
            client=client,
        )
    except NotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"Bucket {output_bucket_name} for upload not found : \
                {str(e)}",
        ) from e
    except Forbidden as e:
        raise HTTPException(
            status_code=403,
            detail=f"Permission error during upload of blob \
                {output_blob_path} in bucket {output_bucket_name} : {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected exception during upload of blob \
                {output_blob_path} in bucket {output_bucket_name} : {str(e)}",
        ) from e
