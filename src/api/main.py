from subprocess import CalledProcessError
import os

import torch
from fastapi import FastAPI, HTTPException
from google.cloud.storage import Client

from src.api.classes.prediction_models import (
    SupportedModel,
    Rgbi15clResnet34UnetModel,
    FlairModel,
)
from src.api.handle_files import download_gcs_folder
from src.api.logger import get_logger
from src.api.setup_flair_configs import setup_config_flair_detect
from src.constants import (
    DATA_FOLDER,
    FLAIR_GCP_PROJECT,
    DEFAULT_FLAIR_CONFIG_DETECT_PATH,
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
    input_image_path: str, model: SupportedModel, output_image_path: str
):
    """Endpoint to execute a Python script with the --conf argument."""
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

    # Setup flair-detect config
    # TODO : download input_image_path from bucket
    setup_config_flair_detect(
        image_path=input_image_path,
        model_path=model_weights_path,
        output_path=output_image_path,
        config_path=DEFAULT_FLAIR_CONFIG_DETECT_PATH,  # TODO : won't work with async calls
    )
    logger.info(f"Config setup for flair-detect for image {input_image_path}")

    # TODO : upload output_image to bucket (separate bucket & prefix in params)
    try:
        use_gpu = torch.cuda.is_available()
        return {
            "message": f"prediction tiff available at {output_image_path}",
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
