from subprocess import CalledProcessError
import os
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from google.cloud.exceptions import NotFound, Forbidden
from google.cloud.storage import Client

from src.api.classes.prediction_models import (
    SupportedModel,
    Rgbi15clResnet34UnetModel,
    FlairModel,
)
from src.api.handle_files import download_gcs_folder, download_file
from src.api.logger import get_logger
from src.api.setup_flair_configs import setup_config_flair_detect
from src.constants import (
    DATA_FOLDER,
    FLAIR_GCP_PROJECT, OUTPUT_FOLDER, INPUT_FOLDER,
)

logger = get_logger()

# Models hash map
available_models: dict[SupportedModel, FlairModel] = {
    SupportedModel.Rgbi15clResnet34Unet: Rgbi15clResnet34UnetModel()
}

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
    # Set identifier for the current prediction (avoid overlap with async calls)
    prediction_id = str(uuid4())

    # Google clood storage client
    client = Client(project=FLAIR_GCP_PROJECT)
    logger.info(f"GCS client created for project {FLAIR_GCP_PROJECT}")

    # Get requested model
    flair_model = available_models[model]
    model_weights_path = os.path.join(
        DATA_FOLDER, flair_model.relative_weights_path
    )

    # Download model from gcs
    if not os.path.exists(model_weights_path):
        download_gcs_folder(
            bucket_name=flair_model.bucket_name,
            blob_prefix=flair_model.blob_prefix,
            local_directory=DATA_FOLDER,
            client=client,
        )
    logger.info(f"Flair model weights available at {model_weights_path}")

    # Create output folder for the prediction
    output_prediction_folder = os.path.join(OUTPUT_FOLDER, prediction_id)
    os.makedirs(output_prediction_folder, exist_ok=True)

    # Download file to process
    image_name = os.path.basename(image_blob_path)
    image_local_path = os.path.join(INPUT_FOLDER, image_name)
    try:
        download_file(bucket_name=image_bucket_name,
                      blob_path=image_blob_path,
                      local_path=image_local_path,
                      client=client)
    except NotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"Blob {image_blob_path} in bucket {image_bucket_name} not found : {str(e)}"
        )
    except Forbidden as e:
        raise HTTPException(
            status_code=403,
            detail=f"Permission error during download of blob {image_blob_path} in bucket {image_bucket_name} : {str(e)}"
        )

    # Setup flair-detect config
    output_name =  os.path.basename(output_blob_path)
    setup_config_flair_detect(
        input_image_path=image_local_path,
        model_weights_path=model_weights_path,
        output_image_name=output_name,
        output_folder=output_prediction_folder,
    )
    logger.info(f"Config setup for flair-detect for model {flair_model.name} and image {image_name}")

    # TODO : upload output_image to bucket
    try:
        use_gpu = torch.cuda.is_available()
        return {
            "prediction_id": prediction_id,
            "message": f"prediction tiff available at {output_prediction_folder}/{output_name}",
            "use_gpu": use_gpu,
        }
        # TODO : uncomment to do the prediction for real
        # result = run(["flair-detect", "--conf", conf_path], check=True, capture_output=True, text=True)
        # return {"message": "Script executed successfully", "output": result.stdout}
    except CalledProcessError as e:
        # Handle script execution errors
        raise HTTPException(
            status_code=500,
            detail=f"Error executing script: {e.stderr.strip()}",
        )
