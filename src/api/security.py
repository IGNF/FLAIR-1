"""This module provides security-related functionalities for authentication.

Auth is done by verifying an HTTP bearer token.
"""

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
    """Validates the provided HTTP bearer token

    The token must be equal to a secret stored in Google Cloud Secret Manager.

    Args:
        credentials : The authorization credentials extracted from
                      the HTTP request.

    Raises:
        HTTPException: If the token is invalid or missing,
                       an HTTP 401 Unauthorized error is raised.

    Returns:
        str: The validated token if the verification succeeds.
    """
    valid_token = retrieve_secret(
        name="FLAIR_API_TOKEN",
        version=1,
        client_secret_manager=SecretManagerServiceClient(),
    )
    if credentials.credentials != valid_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    logger.info("Token is valid")
    return credentials.credentials
