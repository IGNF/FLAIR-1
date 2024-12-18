import os
from pathlib import Path

REPOSITORY_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = os.path.join(REPOSITORY_FOLDER, "data")
INPUT_FOLDER = os.path.join(DATA_FOLDER, "input")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")

DEFAULT_FLAIR_CONFIG_DETECT_PATH = os.path.join(REPOSITORY_FOLDER, "configs", "flair-1-config-detect.yaml")

FLAIR_GCP_PROJECT = os.environ.get("FLAIR_GCP_PROJECT", "netcarbon-datawarehouse")
