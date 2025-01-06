"""This module provides functionalities to run flair detect workflow

From downloading model & image to uploading results.
"""

import os
from time import perf_counter
from subprocess import run

import torch
from google.cloud.storage import Client

from src.api.handle_files import (
    download_gcs_folder,
    download_file,
    upload_file,
)
from src.api.logger import get_logger
from src.api.classes.prediction_models import (
    SupportedModel,
    Rgbi15ClResnet34UnetModel,
    AbstractFlairModel,
)
from src.api.setup_flair_configs import setup_config_flair_detect
from src.constants import (
    DATA_FOLDER,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    FLAIR_GCP_PROJECT,
    FLAIR_DETECT_BATCH_SIZE,
)

logger = get_logger()

# Models hash map
available_models: dict[SupportedModel, AbstractFlairModel] = {
    SupportedModel.Rgbi15clResnet34Unet: Rgbi15ClResnet34UnetModel()
}


def get_output_prediction_folder(prediction_id: str):
    """Returns the path to the output prediction folder.

    Args:
        prediction_id (str): Unique identifier for the prediction.

    Returns:
        str: Path to the output folder for the given prediction ID.
    """
    return os.path.join(OUTPUT_FOLDER, prediction_id)


def get_requested_model(
    model: SupportedModel, client: Client, data_folder: str = DATA_FOLDER
) -> str:
    """Downloads model files from GCS if it's not already present locally.

    Args:
        model (SupportedModel): The model to be downloaded.
        client (Client): Google Cloud Storage client.
        data_folder (str): Path to the local folder where model weights
                           should be stored.

    Returns:
        str: Path to the downloaded model weights.
    """
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

    return model_weights_path


def download_file_to_process(
    image_bucket_name: str,
    image_blob_path: str,
    client: Client,
    input_folder: str = INPUT_FOLDER,
):
    """Downloads an image from a specified GCS bucket to a local input folder.

    Args:
        image_bucket_name (str): Name of the bucket containing the image.
        image_blob_path (str): Path to the image blob within the bucket.
        client (Client): Google Cloud Storage client.
        input_folder (str): Path to the local input folder.

    Returns:
        str: Local path to the downloaded image.
    """
    image_name = os.path.basename(image_blob_path)
    image_local_path = os.path.join(input_folder, image_name)

    download_file(
        bucket_name=image_bucket_name,
        blob_path=image_blob_path,
        local_path=image_local_path,
        client=client,
    )

    logger.info(
        "Blob %s in bucket %s downloaded at %s",
        image_blob_path,
        image_bucket_name,
        image_local_path,
    )

    return image_local_path


def run_prediction(prediction_config_path: str):
    """Runs the flair-detect script using the provided configuration file.

    Args:
        prediction_config_path (str): Path to the flair-detect configuration
                                      file.

    Returns:
        CompletedProcess: Result of the executed prediction process.
    """
    result = run(
        ["flair-detect", "--conf", prediction_config_path],
        check=True,
        capture_output=True,
        text=True,
    )

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
    """Uploads the prediction result file to a specified GCS bucket.

    Args:
        output_prediction_folder (str): Local folder containing the prediction
                                        result.
        output_name (str): Name of the output file.
        output_bucket_name (str): Name of the target Google Cloud Storage
                                  bucket.
        output_blob_path (str): Path to the blob within the bucket.
        client (Client): Google Cloud Storage client.
    """
    output_path = os.path.join(output_prediction_folder, output_name)
    upload_file(
        bucket_name=output_bucket_name,
        blob_path=output_blob_path,
        local_path=output_path,
        client=client,
    )


def flair_detect_service(
    image_bucket_name: str,
    image_blob_path: str,
    model: SupportedModel,
    output_bucket_name: str,
    output_blob_path: str,
    prediction_id: str,
):
    """Main service function that orchestrates the flair-detect workflow.

    Args:
        image_bucket_name (str): Name of the bucket containing the input image.
        image_blob_path (str): Path to the image blob within the bucket.
        model (SupportedModel): The prediction model to be used.
        output_bucket_name (str): Name of the target bucket for the prediction
                                  result.
        output_blob_path (str): Path to the output blob within the bucket.
        prediction_id (str): Unique identifier for the prediction.

    Returns:
        dict: Information about the prediction result, including duration
              and CUDA usage.
    """
    # Create output folder for the prediction
    output_prediction_folder = get_output_prediction_folder(
        prediction_id=prediction_id
    )
    os.makedirs(output_prediction_folder, exist_ok=True)

    # Google clood storage client
    client = Client(project=FLAIR_GCP_PROJECT)

    # Download requested model from netcarbon gcs
    model_weights_path = get_requested_model(model=model, client=client)

    # Download file to process
    image_local_path = download_file_to_process(
        image_bucket_name=image_bucket_name,
        image_blob_path=image_blob_path,
        client=client,
    )
    logger.info("%s ; download image to process done", prediction_id)

    # Setup flair-detect config
    output_name = os.path.basename(output_blob_path)
    prediction_config_path = setup_config_flair_detect(
        input_image_path=image_local_path,
        model_weights_path=model_weights_path,
        output_image_name=output_name,
        output_folder=output_prediction_folder,
        batch_size=FLAIR_DETECT_BATCH_SIZE,
    )
    logger.info("%s ; config setup for flair-detect done", prediction_id)

    # Run the prediction with flair-detect script
    use_gpu = torch.cuda.is_available()
    logger.info("%s cuda is available : %s", prediction_id, use_gpu)

    start_time = perf_counter()
    result = run_prediction(prediction_config_path=prediction_config_path)
    run_prediction_duration = int(round(perf_counter() - start_time))

    logger.info(
        "%s ; run prediction done in %s seconds",
        prediction_id,
        run_prediction_duration,
    )

    # Upload resulted tif to bucket
    upload_result_to_bucket(
        output_prediction_folder=output_prediction_folder,
        output_name=output_name,
        output_bucket_name=output_bucket_name,
        output_blob_path=output_blob_path,
        client=client,
    )

    logger.info("%s ; upload prediction result to bucket", prediction_id)

    return {
        "prediction_id": prediction_id,
        "message": f"prediction tif is available at gs://{output_bucket_name}/{output_blob_path}",
        "cuda_used": use_gpu,
        "result_stdout": result.stdout,
        "run_prediction_duration": run_prediction_duration,
    }
