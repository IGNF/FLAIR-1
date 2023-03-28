import os
import numpy as np
from os.path import join
from pathlib import Path
import json
import random
from random import shuffle
import re
import yaml
try:
  from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
  from pytorch_lightning.utilities.rank_zero import rank_zero_only  

def load_data(paths_data, val_percent=0.8, use_metadata=True):
    
    def _gather_data(domains, paths_data: dict, use_metadata: bool, test_set: bool) -> dict:

        #### return data paths
        def get_data_paths (path, filter):
            for path in Path(path).rglob(filter):
                 yield path.resolve().as_posix()        

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

        data = {'IMG':[],'MSK':[],'MTD':[]}   
        for domain in domains: 
            for area in os.listdir(Path(paths_data['path_aerial_'+['train' if test_set == False else 'test'][0]], domain)):
                data['IMG'] += sorted(list(get_data_paths(Path(paths_data['path_aerial_'+['train' if test_set == False else 'test'][0]], domain, area),'IMG*.tif')),
                                      key=lambda x: int(x.split('_')[-1][:-4]),
                                     )
                if test_set == False:
                    data['MSK'] += sorted(list(get_data_paths(Path(paths_data['path_labels_'+['train' if test_set == False else 'test'][0]], domain, area),'MSK*.tif')),
                                          key=lambda x: int(x.split('_')[-1][:-4]),
                                         )        
        if use_metadata == True:

            with open(paths_data['path_metadata_aerial'], 'r') as f:
                metadata_dict = json.load(f)              
            for img in data['IMG']:
                curr_img     = img.split('/')[-1][:-4]
                enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
                enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
                enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
                mtd_enc      = enc_coords+enc_alti+enc_camera+enc_temporal 
                data['MTD'].append(mtd_enc)

        if test_set == False:
            if len(data['IMG']) != len(data['MSK']): 
                print('[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging.')
            if data['IMG'][0][-10:-4] != data['MSK'][0][-10:-4] or data['IMG'][-1][-10:-4] != data['MSK'][-1][-10:-4]: 
                print('[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging.')                

        return data
    
    
    path_trainval = Path(paths_data['path_aerial_train'])
    trainval_domains = os.listdir(path_trainval)
    shuffle(trainval_domains)
    idx_split = int(len(trainval_domains) * val_percent)
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 
    
    dict_train = _gather_data(train_domains, paths_data, use_metadata=use_metadata, test_set=False)
    dict_val = _gather_data(val_domains, paths_data, use_metadata=use_metadata, test_set=False)
    
    path_test = Path(paths_data['path_aerial_test'])
    test_domains = os.listdir(path_test)
    
    dict_test = _gather_data(test_domains, paths_data, use_metadata=use_metadata, test_set=True)
    
    return dict_train, dict_val, dict_test

def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
@rank_zero_only
def step_loading(paths_data, use_metadata: bool) -> dict:
    print('+'+'-'*29+'+', '   LOADING DATA   ', '+'+'-'*29+'+')
    train, val, test = load_data(paths_data, use_metadata=use_metadata)
    return train, val, test    
   
@rank_zero_only
def print_recap(config, dict_train, dict_val, dict_test):
    print('\n+'+'='*80+'+', 'Model name: ' + config['outputs']["out_model_name"], '+'+'='*80+'+', f"{'[---TASKING---]'}", sep='\n')
    for info, val in zip(["use weights", "use metadata", "use augmentation"], [config["use_weights"], config["use_metadata"], config["use_augmentation"]]): 
        print(f"- {info:25s}: {'':3s}{val}")
    print('\n+'+'-'*80+'+', f"{'[---DATA SPLIT---]'}", sep='\n')
    for split_name, d in zip(["train", "val", "test"], [dict_train, dict_val, dict_test]): 
        print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    print('\n+'+'-'*80+'+', f"{'[---HYPER-PARAMETERS---]'}", sep='\n')
    for info, val in zip(["batch size", "learning rate", "epochs", "nodes", "GPU per nodes", "accelerator", "workers"], [config["batch_size"], config["learning_rate"], config["num_epochs"], config["num_nodes"], config["gpus_per_node"], config["accelerator"], config["num_workers"]]): 
        print(f"- {info:25s}: {'':3s}{val}")        
    print('\n+'+'-'*80+'+', '\n')

@rank_zero_only
def print_metrics(miou, ious):
    classes = ['building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous',
               'brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
    print('\n')
    print('-'*40)
    print(' '*8, 'Model mIoU : ', round(miou, 4))
    print('-'*40)
    print ("{:<25} {:<15}".format('Class','iou'))
    print('-'*40)
    for k, v in zip(classes, ious):
        print ("{:<25} {:<15}".format(k, v))
    print('\n\n')
