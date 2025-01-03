from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from google.cloud.secretmanager import SecretManagerServiceClient

from src.api.logger import get_logger
from src.api.utils import retrieve_secret


logger = get_logger()

security = HTTPBearer()


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    valid_token = retrieve_secret(
        name="FLAIR_API_TOKEN",
        version=1,
        client_secret_manager=SecretManagerServiceClient(),
    )
    if credentials.credentials != valid_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    logger.info("Token is valid")
    return credentials.credentials
