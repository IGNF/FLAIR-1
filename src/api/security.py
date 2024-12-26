from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from google.cloud.secretmanager import SecretManagerServiceClient

from src.api.logger import get_logger
from src.api.utils import retrieve_secret


logger = get_logger()

VALID_TOKEN = retrieve_secret(
    name="FLAIR_API_TOKEN",
    version=1,
    client_secret_manager=SecretManagerServiceClient(),
)

security = HTTPBearer()


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    logger.info("Token is valid")
    return credentials.credentials
