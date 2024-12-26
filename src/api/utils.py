from google.cloud.secretmanager import SecretManagerServiceClient

from src.api.logger import get_logger


logger = get_logger()


def retrieve_secret(
    name: str, version: int, client_secret_manager: SecretManagerServiceClient
):
    """This function retrieve secret from GCP

    Args:
        name: name of the secret
        version: version of the secret
        client_secret_manager: client to retrieve secrets from GCP
    Returns:
        secret : secret as data

    """
    secret_name = (
        "projects/{project_id}/secrets/{name}/versions/{version}".format(
            name=name, version=version, project_id="247332828466"
        )
    )
    try:
        response = client_secret_manager.access_secret_version(
            name=secret_name
        )
        return response.payload.data.decode("UTF-8")
    except NameError:
        logger.info("Error: secret not found %s", secret_name)
        return None
