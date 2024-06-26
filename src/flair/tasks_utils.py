import os, json
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A

from src.flair.model import FLAIR_ModelFactory
from src.flair.data_module import flair_datamodule
from src.flair.task_module import segmentation_task_training, segmentation_task_predict



def get_data_module(config, 
                    dict_train : dict =None, 
                    dict_val : dict = None, 
                    dict_test : dict = None,
                    ):
    """
    This function creates a data module for training, validation, and testing.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for the data module.
    dict_train (dict): Dictionary containing training data.
    dict_val (dict): Dictionary containing validation data.
    dict_test (dict): Dictionary containing test data.

    Returns:
    dm: Data module with specified configuration.
    """
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(config["use_augmentation"], bool), "use_augmentation must be a boolean"
    assert isinstance(config["use_metadata"], bool), "use_metadata must be a boolean"   
    
    if config["use_augmentation"]:
        transform_set = A.Compose([A.VerticalFlip(p=0.5),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomRotate90(p=0.5)]
        )
    else:
        transform_set = None

    dm = flair_datamodule(
        dict_train = dict_train,
        dict_val = dict_val,
        dict_test = dict_test,
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        drop_last = True,
        num_classes = len(config["classes"]),
        channels = config["channels"],
        use_metadata = config["use_metadata"],
        use_augmentations = transform_set,
        norm_type = config["norm_type"],
        means = config["norm_means"],
        stds = config["norm_stds"]
    )
    
    return dm



def get_segmentation_module(config, 
                            stage : str = 'train',
                            ):
    """
    This function creates a segmentation module for training or prediction.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for the segmentation module.
    stage (str): Stage for which the segmentation module is created ('train' or 'predict').

    Returns:
    seg_module: Segmentation module with specified configuration.
    """
    assert stage in ['train', 'predict'], "stage must be either 'train' or 'predict'"
    
    # Define model
    model = FLAIR_ModelFactory(architecture = config['model_architecture'],
                         encoder = config['encoder_name'],
                         n_channels = len(config["channels"]), 
                         n_classes = len(config["classes"]), 
                         use_metadata = config["use_metadata"],
    )

    #print(model)

    if stage == 'train':
        if config["use_weights"]:
            with torch.no_grad():
                class_weights = torch.FloatTensor([config["classes"][i][0] for i in config["classes"]])
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

        scheduler = ReduceLROnPlateau(
            optimizer = optimizer,
            mode = "min",
            factor = 0.5,
            patience = 10,
            cooldown = 4,
            min_lr = 1e-7,
        )

        seg_module = segmentation_task_training(
            model = model,
            class_infos = config["classes"],
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            use_metadata=config["use_metadata"],
        )

    elif stage == 'predict':
        seg_module = segmentation_task_predict(
            model = model,
            num_classes = len(config["classes"]),
            use_metadata = config["use_metadata"],
        )        

    return seg_module




def gather_paths(config, split='train'):
    if split == 'train':
        if config['paths']['train_csv'] is not None and os.path.isfile(config['paths']['train_csv']) and config['paths']['train_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['train_csv'], header=None)
        else:
            print("Invalid .csv training file path.")
            raise SystemExit()
    elif split == 'val':
        if config['paths']['val_csv'] is not None and os.path.isfile(config['paths']['val_csv']) and config['paths']['val_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['val_csv'], header=None)
        else:
            print("Invalid .csv val file path.")
            raise SystemExit()
    elif split == 'test':
        if config['paths']['test_csv'] is not None and os.path.isfile(config['paths']['test_csv']) and config['paths']['test_csv'].endswith('.csv'):
            paths = pd.read_csv(config['paths']['test_csv'], header=None)
        else:
            print("Invalid .csv test file path.") 
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