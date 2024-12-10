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
        blob_prefix (str): The prefix or folder path in the bucket to copy from.
        local_directory (str): Path to the local directory where files should be copied.
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
