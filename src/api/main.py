import gc
from subprocess import CalledProcessError
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from google.cloud.exceptions import NotFound, Forbidden

from src.api.classes.prediction_models import SupportedModel
from src.api.flair_detect_service import (
    flair_detect_service,
)
from src.api.security import verify_token
from src.api.logger import get_logger

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
    token: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
):
    try:
        result = flair_detect_service(
            image_bucket_name=image_bucket_name,
            image_blob_path=image_blob_path,
            model=model,
            output_bucket_name=output_bucket_name,
            output_blob_path=output_blob_path,
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
        # Force garbage collection to avoid future exceptions
        gc.collect()

    return result
