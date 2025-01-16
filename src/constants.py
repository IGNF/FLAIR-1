import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPOSITORY_FOLDER = Path(__file__).parent.parent
DATA_FOLDER = os.path.join(REPOSITORY_FOLDER, "data")
INPUT_FOLDER = os.path.join(DATA_FOLDER, "input")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
DEFAULT_FLAIR_CONFIG_DETECT_PATH = os.path.join(
    REPOSITORY_FOLDER, "configs", "flair-1-config-detect.yaml"
)

FLAIR_GCP_PROJECT = os.environ.get(
    "FLAIR_GCP_PROJECT", "netcarbon-datawarehouse"
)

FLAIR_DETECT_BATCH_SIZE = int(os.environ.get("FLAIR_DETECT_BATCH_SIZE", 4))
GARBAGE_COLLECTOR_FREQUENCY = int(
    os.environ.get("GARBAGE_COLLECTOR_FREQUENCY", 10)
)

clean = os.environ.get("CLEAN_ALL_FILES_AFTER_PREDICTION", "false")
CLEAN_ALL_FILES_AFTER_PREDICTION = True if clean.lower() == "true" else False
