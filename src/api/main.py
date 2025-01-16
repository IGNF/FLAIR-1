"""Main module to start the API application"""

import gc
import shutil
from subprocess import CalledProcessError
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from google.cloud.exceptions import NotFound, Forbidden

from src.api.classes.prediction_models import SupportedModel
from src.api.flair_detect_service import flair_detect_service
from src.api.security import verify_token
from src.api.logger import get_logger
from src.constants import (
    CLEAN_ALL_FILES_AFTER_PREDICTION,
    OUTPUT_FOLDER,
    INPUT_FOLDER,
)

logger = get_logger()


# FastAPI app instance
app = FastAPI(title="FLAIR-1 API")


@app.post("/flair-detect")
async def flair_detect(
    image_bucket_name: str,
    image_blob_path: str,
    model: SupportedModel,
    output_bucket_name: str,
    output_blob_path: str,
    prediction_id: str,
    token: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
):
    """Handles POST requests to the /flair-detect endpoint.

    Args:
        image_bucket_name (str): Name of the Google Cloud Storage bucket
                                 containing the input image.
        image_blob_path (str): Path to the image blob within the bucket.
        model (SupportedModel): The model to be used for prediction.
        output_bucket_name (str): Name of the Google Cloud Storage bucket to
                                  store the prediction result.
        output_blob_path (str): Path to the blob within the output bucket
                                where the result will be stored.
        prediction_id (str): Unique identifier for the prediction.
        token: Bearer token for authentication.

    Returns:
            dict: Result of the prediction including details such as
                  prediction ID, CUDA usage, and execution duration.

    Raises:
            HTTPException: Raised when an execution error occurs.
    """
    try:
        result = flair_detect_service(
            image_bucket_name=image_bucket_name,
            image_blob_path=image_blob_path,
            model=model,
            output_bucket_name=output_bucket_name,
            output_blob_path=output_blob_path,
            prediction_id=prediction_id,
        )
    except NotFound as e:
        raise HTTPException(
            status_code=404,
            detail=f"{str(e)}",
        ) from e
    except Forbidden as e:
        raise HTTPException(
            status_code=403,
            detail=f"{str(e)}",
        ) from e
    except CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing flair-detect script: {e.stderr.strip()}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{str(e)}",
        ) from e
    finally:
        # Clean all input / output files to avoid memory leak if requested
        if CLEAN_ALL_FILES_AFTER_PREDICTION:
            shutil.rmtree(INPUT_FOLDER, ignore_errors=True)
            shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        # Force garbage collection to avoid future exceptions
        gc.collect()

    return result
