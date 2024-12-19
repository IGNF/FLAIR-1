import os
from time import perf_counter
from uuid import uuid4

import torch
from fastapi import FastAPI
from google.cloud.storage import Client

from src.api.classes.prediction_models import SupportedModel
from src.api.flair_detect_service import (
    get_requested_model,
    download_file_to_process,
    run_prediction,
    upload_result_to_bucket,
)
from src.api.logger import get_logger
from src.api.setup_flair_configs import setup_config_flair_detect
from src.constants import FLAIR_GCP_PROJECT, OUTPUT_FOLDER


logger = get_logger()


# FastAPI app instance
app = FastAPI(title="FLAIR-1 API")


# Endpoint for flair-detect
@app.post("/flair-detect")
async def flair_detect(
    image_bucket_name: str,
    image_blob_path: str,
    model: SupportedModel,
    output_bucket_name: str,
    output_blob_path: str,
):
    # Set identifier for the current prediction (avoid overlap with async calls
    prediction_id = str(uuid4())

    # Create output folder for the prediction
    output_prediction_folder = os.path.join(OUTPUT_FOLDER, prediction_id)
    os.makedirs(output_prediction_folder, exist_ok=True)

    # Google clood storage client
    client = Client(project=FLAIR_GCP_PROJECT)

    # Download requested model from netcarbon gcs
    flair_model, model_weights_path = get_requested_model(
        model=model, client=client
    )

    # Download file to process
    start_time = perf_counter()
    image_local_path = download_file_to_process(
        image_bucket_name=image_bucket_name,
        image_blob_path=image_blob_path,
        client=client,
    )
    download_file_duration = int(round(perf_counter() - start_time))

    logger.info("download file duration: %s seconds", download_file_duration)

    # Setup flair-detect config
    output_name = os.path.basename(output_blob_path)
    prediction_config_path = setup_config_flair_detect(
        input_image_path=image_local_path,
        model_weights_path=model_weights_path,
        output_image_name=output_name,
        output_folder=output_prediction_folder,
    )
    logger.info(
        "Config setup for flair-detect for model %s and image %s",
        flair_model.name,
        image_blob_path,
    )

    # Run the prediction with flair-detect script
    use_gpu = torch.cuda.is_available()
    logger.info("cuda is available : %s", use_gpu)

    start_time = perf_counter()
    result = run_prediction(prediction_config_path=prediction_config_path)
    run_prediction_duration = int(round(perf_counter() - start_time))

    logger.info("run prediction duration: %s seconds", run_prediction_duration)

    # Upload resulted tif to bucket
    start_time = perf_counter()
    upload_result_to_bucket(
        output_prediction_folder=output_prediction_folder,
        output_name=output_name,
        output_bucket_name=output_bucket_name,
        output_blob_path=output_blob_path,
        client=client,
    )
    upload_result_duration = int(round(perf_counter() - start_time))

    logger.info("upload result duration: %s seconds", upload_result_duration)

    return {
        "prediction_id": prediction_id,
        "message": f"prediction tif is available at gs://{output_bucket_name}/{output_blob_path}",
        "cuda_used": use_gpu,
        "result_stdout": result.stdout,
        "perf": {
            "download_file_duration": download_file_duration,
            "run_prediction_duration": run_prediction_duration,
            "upload_result_duration": upload_result_duration,
        },
    }
