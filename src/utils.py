import os
import numpy as np
import logging
import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from pytorch_lightning.utilities.rank_zero import rank_zero_only  

def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
     

def setup_logger(config):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(Path(config['paths']["out_folder"], config['paths']["out_model_name"], 'train.log').as_posix())  # File handler
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    #format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    format = logging.Formatter('%(message)s')
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    lning_logger = logging.getLogger('pytorch_lightning')
    lning_logger.addHandler(f_handler) 

    logging.info(datetime.now().strftime("Starting : %Y-%m-%d  %H:%M")+'\n')



def gather_paths(config, split='train'):
    if split == 'train':
        if config['paths']['train_csv'] is not None and os.path.isfile(config['paths']['train_csv']) and config['paths']['train_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['train_csv'], header=None)
        else:
            logging.error("Invalid .csv training file path.")
            raise SystemExit()
    elif split == 'val':
        if config['paths']['val_csv'] is not None and os.path.isfile(config['paths']['val_csv']) and config['paths']['val_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['val_csv'], header=None)
        else:
            logging.error("Invalid .csv val file path.")
            raise SystemExit()
    elif split == 'test':
        if config['paths']['test_csv'] is not None and os.path.isfile(config['paths']['test_csv']) and config['paths']['test_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['test_csv'], header=None)
        else:
            logging.error("Invalid .csv test file path.") 
            raise SystemExit()       
        
    images = paths.iloc[:,0].tolist()
    labels = paths.iloc[:,1].tolist()
    if config['use_metadata'] == True:
        metadata = parsing_metadata(images, config)
    else:
        metadata = []
    dict_paths = {'IMG':images,'MSK':labels,'MTD':metadata}       
    
    return dict_paths


def parsing_metadata(image_path_list, config):
    #### encode metadata
    def coordenc_opt(coords, enc_size=32) -> np.array:
        d = int(enc_size/2)
        d_i = np.arange(0, d / 2)
        freq = 1 / (10e7 ** (2 * d_i / d))

        x,y = coords[0]/10e7, coords[1]/10e7
        enc = np.zeros(d * 2)
        enc[0:d:2]    = np.sin(x * freq)
        enc[1:d:2]    = np.cos(x * freq)
        enc[d::2]     = np.sin(y * freq)
        enc[d + 1::2] = np.cos(y * freq)
        return list(enc)           

    def norm_alti(alti: int) -> float:
        min_alti = 0
        max_alti = 3164.9099121094
        return [(alti-min_alti) / (max_alti-min_alti)]        

    def format_cam(cam: str) -> np.array:
        return [[1,0] if 'UCE' in cam else [0,1]][0]

    def cyclical_enc_datetime(date: str, time: str) -> list:
        def norm(num: float) -> float:
            return (num-(-1))/(1-(-1))
        year, month, day = date.split('-')
        if year == '2018':   enc_y = [1,0,0,0]
        elif year == '2019': enc_y = [0,1,0,0]
        elif year == '2020': enc_y = [0,0,1,0]
        elif year == '2021': enc_y = [0,0,0,1]    
        sin_month = np.sin(2*np.pi*(int(month)-1/12)) ## months of year
        cos_month = np.cos(2*np.pi*(int(month)-1/12))    
        sin_day = np.sin(2*np.pi*(int(day)/31)) ## max days
        cos_day = np.cos(2*np.pi*(int(day)/31))     
        h,m=time.split('h')
        sec_day = int(h) * 3600 + int(m) * 60
        sin_time = np.sin(2*np.pi*(sec_day/86400)) ## total sec in day
        cos_time = np.cos(2*np.pi*(sec_day/86400))
        return enc_y+[norm(sin_month),norm(cos_month),norm(sin_day),norm(cos_day),norm(sin_time),norm(cos_time)]     
    
    
    with open(config['paths']['path_metadata_aerial'], 'r') as f:
        metadata_dict = json.load(f)  
    
    MTD = []
    for img in image_path_list:
        curr_img     = img.split('/')[-1][:-4]
        enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
        enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
        enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
        enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
        mtd_enc      = enc_coords+enc_alti+enc_camera+enc_temporal 
        MTD.append(mtd_enc)  
        
    return MTD


@rank_zero_only
def print_recap_training(config, dict_train=None, dict_val=None, dict_test=None):
    logging.info('\n+' + '=' * 80 + '+')
    logging.info('Model name: ' + config['paths']["out_model_name"])
    logging.info('+' + '=' * 80 + '+')
    logging.info("[---TASKING---]")
    for info, val in zip(["model architecture", "encoder name", "use weights", "use metadata", "use augmentation"], 
                         [config['model_architecture'], config['encoder_name'], config["use_weights"], config["use_metadata"], config["use_augmentation"]]): 
        logging.info(f"- {info:25s}: {'':3s}{val}")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---DATA SPLIT---]')
    if config['tasks']['train']:
        for split_name, d in zip(["train", "val"], [dict_train, dict_val]): 
            logging.info(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    if config['tasks']['predict']:
        logging.info(f"- {'test':25s}: {'':3s}{len(dict_test['IMG'])} samples")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---HYPER-PARAMETERS---]')
    for info, val in zip(["batch size", "learning rate", "seed", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], 
                         [config["batch_size"], config["learning_rate"], config["seed"], config["num_epochs"], config["num_nodes"], config["gpus_per_node"], config["accelerator"], config["num_workers"]]): 
        logging.info(f"- {info:25s}: {'':3s}{val}")        
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---NORMALIZATION---]')
    logging.info(f"- norm type: {config['norm_type']}")
    logging.info(f"- norm means: {config['norm_means']}")
    logging.info(f"- norm stds: {config['norm_stds']}")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---PREDICT CONFIG---]')
    logging.info(f"- georeferencing output: {config['georeferencing_output']}")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---COMPUTATIONAL RESOURCES---]')
    logging.info(f"- accelerator: {config['accelerator']}")
    logging.info(f"- num nodes: {config['num_nodes']}")
    logging.info(f"- gpus per node: {config['gpus_per_node']}")
    logging.info(f"- strategy: {config['strategy']}")
    logging.info(f"- num workers: {config['num_workers']}")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---PRINT PROGRESS---]')
    logging.info(f"- enable progress bar: {config['enable_progress_bar']}")
    logging.info(f"- progress rate: {config['progress_rate']}")
    logging.info('+' + '-' * 80 + '+')
    logging.info('[---CLASSES AND WEIGHTS---]')
    for class_id, class_info in config['classes'].items():
        weight, name = class_info
        logging.info(f"- Class {class_id} ({name}): weight {weight}")
    logging.info('+' + '-' * 80 + '+\n\n')



@rank_zero_only
def print_save_metrics(config, out_folder, confmat, ious, miou):

    out_folder_metrics = Path('/'.join(out_folder.as_posix().split('/')[:-1]), 'metrics')
    out_folder_metrics.mkdir(exist_ok=True, parents=True)

    np.save(out_folder_metrics.as_posix()+'/confmat.npy', confmat)

    logging.info('\n'+'-'*65)
    logging.info(' '*20+'Model mIoU : '+str(round(miou, 4)))
    logging.info('-'*65)
    logging.info ("{:<25} {:<15} {:<10}".format('Class','Weight', 'IoU'))
    logging.info('-'*65)
    for iou, class_info in zip(ious, config['classes'].items()):
        _ , info = class_info
        logging.info("{:<25} {:<15} {:<10}".format(info[1], info[0], iou))
    logging.info('\n\n') 