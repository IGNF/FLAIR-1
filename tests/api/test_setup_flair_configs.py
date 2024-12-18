import os.path

import yaml

from src.api.setup_flair_configs import setup_config_flair_detect


def test_setup_config_flair_detect(
        tests_output_folder
):
    # init
    input_image_path = "/data/input/6939be99-cb84-47a2-ae31-96e1f37a24a9/75-2018-0645-6865-LA93-0M20-E080_RGBN.tif"
    model_weights_path = "75-2018-0645-6865-LA93-0M20-E080_rgbi_15cl_resnet34-unet.tif"
    output_image_name = "75-2018-0645-6865-LA93-0M20-E080_rgbi_15cl_resnet34-unet.tif"
    output_folder = tests_output_folder

    # act
    runtime_config_path = setup_config_flair_detect(
        input_image_path=input_image_path,
        model_weights_path=model_weights_path,
        output_image_name=output_image_name,
        output_folder=output_folder
    )

    # assert
    assert os.path.exists(runtime_config_path)
    with open(runtime_config_path) as f:
        updated_config = yaml.safe_load(f)

    assert updated_config["output_path"] == output_folder
    assert updated_config["output_name"] == output_image_name
    assert updated_config["input_img_path"] == input_image_path
    assert updated_config["channels"] == [1, 2, 3, 4]
    assert updated_config["model_weights"] == model_weights_path
    assert updated_config["img_pixels_detection"] == 1024
    assert updated_config["margin"] == 256
    assert updated_config["num_worker"] == 0
