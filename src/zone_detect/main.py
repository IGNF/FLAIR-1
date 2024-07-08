import os
import logging 
import warnings
import torch
import rasterio
import yaml

from pathlib import Path
import argparse 
from torch.utils.data import DataLoader
from rasterio.features import geometry_window
from tqdm import tqdm

from src.zone_detect.slicing_job import slice_extent, create_polygon_from_bounds
from src.zone_detect.model import load_model
from src.zone_detect.dataset import Sliced_Dataset, convert


warnings.simplefilter(action='ignore', category=FutureWarning)

#### CONF FILE
argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="Path to the .yaml config file")

#### LOGGERS 
LOGGER = logging.getLogger(__name__)

log = logging.getLogger('stdout_detection')
log.setLevel(logging.DEBUG)
STD_OUT_LOGGER = log

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
STD_OUT_LOGGER.addHandler(ch)



def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)



def setup(args):
    config = read_config(args.conf)
    use_gpu = (False if torch.cuda.is_available() is False else config['use_gpu'])
    device = torch.device("cuda" if use_gpu else "cpu")

    assert isinstance(config['output_path'], str), "Output path does not exist."
    assert os.path.exists(config['input_img_path']), "Input image path does not exist."
    assert config['margin'] * 2 < config['img_pixels_detection'], "Margin is too large : margin*2 < img_pixels_detection"
    assert config['output_type'] in ['class_prob', 'argmax'], "Invalid output type: should be argmax or class_prob."
    assert config['norma_task'][0]['norm_type'] in ['custom', 'scaling'], "Invalid normalization type: should be custom or scaling."
    assert os.path.isfile(config['model_weights']), "Model weights file does not exist."
    if not config['output_name'].endswith('.tif'):
        config['output_name'] += '.tif'  

    try:
        Path(config['output_path']).mkdir(parents=True, exist_ok=True)
        path_out = os.path.join(config['output_path'], config['output_name'])
        if os.path.exists(path_out):
            os.remove(path_out)  # Removing if existing
        return config, path_out, device, use_gpu

    except Exception as error:
        print(f"Something went wrong during detection configuration: {error}")





def conf_log(config, resolution):
    print(f"""
    |- output path: {config['output_path']}
    |- output raster name: {config['output_name']}

    |- input image path: {config['input_img_path']}
    |- channels: {config['channels']}
    |- resolution: {resolution}\n
    |- image size for detection: {config['img_pixels_detection']}
    |- overlap margin: {config['margin']}
    |- write dataframe: {config['write_dataframe']}
    |- number of classes: {config['n_classes']}
    |- normalization: {config['norma_task'][0]['norm_type']}
    |- output type: {config['output_type']}\n
    |- model weights path: {config['model_weights']}
    |- model arch: {config['model_name']}
    |- encoder: {config['encoder_name']}
    |- model template: {config['model_framework']['model_provider']}
    |- device: {"cuda" if config['use_gpu'] else "cpu"}
    |- batch size: {config['batch_size']}\n\n""")       



def prepare(config, device):
    
    STD_OUT_LOGGER.info(f"""
    ##############################################
    ZONE DETECTION
    ##############################################

    CUDA available? {torch.cuda.is_available()}""")

    ## slicing extent for overlapping detection 
    sliced_dataframe, profile, resolution = slice_extent(in_img=config['input_img_path'],
                                                                patch_size=config['img_pixels_detection'],  
                                                                margin=config['margin'], 
                                                                output_name=config['output_name'],
                                                                output_path=config['output_path'],
                                                                write_dataframe=config['write_dataframe'],
                                                               )
    ## log
    conf_log(config, resolution)
    STD_OUT_LOGGER.info(f"""    [x] sliced input raster to {len(sliced_dataframe)} squares...""")
    ## loading model and weights
    model = load_model(config)
    model.eval()
    model = model.to(device)  
    STD_OUT_LOGGER.info(f"""    [x] loaded model and weights...""")

    return sliced_dataframe, profile, resolution, model
    



def main():

    # reading yaml
    args = argParser.parse_args()
    config, path_out, device, use_gpu = setup(args)

    input_img_path = config['input_img_path']
    channels = config['channels']
    img_pixels_detection = config['img_pixels_detection']
    norma_task = config['norma_task']
    batch_size = config['batch_size']    
    num_worker = config['num_worker']
    output_type = config['output_type']    
    margin = config['margin']
    n_classes = config['n_classes']

    # slicing and model gathering
    sliced_dataframe, profile, resolution, model = prepare(config, device)
    
    # get dataset 
    dataset = Sliced_Dataset(dataframe=sliced_dataframe,
                            img_path=input_img_path,
                            resolution=resolution,
                            bands=channels, 
                            patch_detection_size=img_pixels_detection,
                            norma_dict=norma_task,
                            )    
    
    # prepare output raster
    out_overall_profile = profile.copy()
    out_overall_profile.update({'dtype':'uint8', 'compress':'LZW', 'driver':'GTiff', 'BIGTIFF':'YES', 'tiled':True, 
                                'blockxsize':img_pixels_detection, 'blockysize':img_pixels_detection})
    out_overall_profile['count'] = [1 if output_type == 'argmax' else n_classes][0]
    out = rasterio.open(path_out, 'w+', **out_overall_profile)   
    
    # get Dataloader
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_worker,
                             pin_memory=True,
                             )
    # inference loop
    STD_OUT_LOGGER.info(f"""    [ ] starting inference...\n""")
    for samples in tqdm(data_loader):
        imgs = samples["image"]
        if use_gpu:
            imgs = imgs.cuda()
        with torch.no_grad():
            logits = model(imgs)
            if config['model_framework']['model_provider'] == 'HuggingFace':
                logits = logits.logits
            logits.to(device)
        predictions = torch.softmax(logits, dim=1)
        predictions = predictions.cpu().numpy()
        indices = samples["index"].cpu().numpy()    

        # writing windowed raster to output rastert 
        for prediction, index in zip(predictions, indices):
            # removing margins
            prediction = prediction[:,0+margin:img_pixels_detection-margin,0+margin:img_pixels_detection-margin]
            prediction = convert(prediction, output_type)
            sliced_patch_bounds = create_polygon_from_bounds(sliced_dataframe.at[index[0], 'left'], sliced_dataframe.at[index[0], 'right'], 
                                                             sliced_dataframe.at[index[0], 'bottom'], sliced_dataframe.at[index[0], 'top'])
            window = geometry_window(out, [sliced_patch_bounds], pixel_precision=6)
            window = window.round_shape(op='ceil', pixel_precision=4)
            #write
            if output_type == "argmax":
                out.write(prediction, window=window)
            else:
                out.write_band([i for i in range(1, n_classes + 1)], prediction, window=window)      
                
    out.close()
    dataset.close_raster()
    STD_OUT_LOGGER.info(f"""    
                        
    [X] done writing to {path_out.split('/')[-1]} raster file.\n""")


if __name__ == '__main__':
    main()         





    
