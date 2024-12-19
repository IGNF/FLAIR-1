import os

from pathlib import Path
from google.cloud.storage import Client

from src.api.logger import get_logger

logger = get_logger()


def download_gcs_folder(
    bucket_name: str, blob_prefix: str, local_directory: str, client: Client
):
    """Recursively copy files from a GCS bucket to a local directory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_prefix (str): The prefix or folder path in the bucket to copy from
        local_directory (str): Path to the local directory where files should
                               be copied.
        client: gcs Client
    """
    # Ensure local directory exists
    Path(local_directory).mkdir(parents=True, exist_ok=True)

    # List blobs in the specified prefix
    blobs = client.list_blobs(bucket_name, prefix=blob_prefix)

    for blob in blobs:
        # Skip directory placeholders
        if blob.name.endswith("/"):
            logger.info(f"{blob.name} skipped directory placeholders")
            continue

        # Define local file path
        local_file_path = os.path.join(local_directory, blob.name)

        # Ensure local subdirectories exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob to the local file
        blob.download_to_filename(local_file_path)

        logger.info(f"Copied: {blob.name} to {local_file_path}")


def download_file(
    bucket_name: str,
    blob_path: str,
    local_path: str,
    client: Client,
    force=False,
):
    """Download a file from bucket to the dedicated local path

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_path (str): blob path in the bucket
        local_path (str): where to save the file
        client: gcs Client
        force (bool): force download even if file already exists
    """
    if not os.path.exists(local_path) or force:
        # Get the bucket and the blob
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Ensure local subdirectories exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download
        blob.download_to_filename(local_path)


def upload_file(
    bucket_name: str, blob_path: str, local_path: str, client: Client
):
    """Upload a file to bucket to the dedicated blob path

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_path (str): blob path in the bucket
        local_path (str): the file to upload
        client: gcs Client
    """
    # Get the bucket and the blob
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Download
    blob.upload_from_filename(local_path)
