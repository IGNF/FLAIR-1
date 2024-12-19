import os

import yaml

from src.constants import DEFAULT_FLAIR_CONFIG_DETECT_PATH


def setup_config_flair_detect(
    input_image_path: str,
    model_weights_path: str,
    output_image_name: str,
    output_folder: str,
):
    """Setup the configuration file for the prediction.

    Args:
        input_image_path: The path to the input image.
        model_weights_path: The path to the model weights.
        output_image_name: The name of the output image.
        output_folder: Folder where to store config and output_image.

    Returns:
        the path to the generated conf file
    """
    with open(DEFAULT_FLAIR_CONFIG_DETECT_PATH) as f:
        flair = yaml.safe_load(f)
        flair["output_path"] = output_folder
        flair["output_name"] = output_image_name
        flair["input_img_path"] = input_image_path
        flair["channels"] = [1, 2, 3, 4]
        flair["model_weights"] = model_weights_path
        flair["img_pixels_detection"] = 1024
        flair["margin"] = 256
        flair["num_worker"] = 0

    runtime_config_path = os.path.join(
        output_folder, "flair-1-config-detect.yaml"
    )

    with open(runtime_config_path, "w+") as f:
        yaml.dump(
            flair,
            f,
            default_flow_style=None,
            sort_keys=False,
        )

    return runtime_config_path
