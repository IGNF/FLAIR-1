import os

import yaml


def setup_config_flair_detect(
    image_path: str,
    model_path: str,
    output_path: str,
    config_path: str,
):
    """Setup the configuration file for the prediction.

    Args:
        image_path: The path to the image.
        output_path: The path to the output.
        config_path: The path to the configuration file.
        model_path: The path to the model.
    """
    output_name = os.path.basename(output_path)
    output_path = os.path.dirname(output_path)
    with open(config_path) as f:
        flair = yaml.safe_load(f)
        flair["output_path"] = output_path
        flair["output_name"] = output_name
        flair["input_img_path"] = image_path
        flair["channels"] = [1, 2, 3, 4]
        flair["model_weights"] = model_path
        flair["img_pixels_detection"] = 1024
        flair["margin"] = 256
        flair["num_worker"] = 0

        yaml.dump(
            flair,
            open(config_path, "w+"),
            default_flow_style=None,
            sort_keys=False,
        )
