#general 
import os
import numpy as np
import yaml
import json
import random
from pathlib import Path
import argparse 

#deep learning
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything 

import albumentations as A

#flair-one baseline modules 
from py_module.utils import load_data, step_loading, read_config, print_recap, print_metrics
from py_module.datamodule import OCS_DataModule
from py_module.model import SMP_Unet_meta
from py_module.task_module import SegmentationTask
from py_module.writer import PredictionWriter
from py_module.generate_miou import generate_miou

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")

def get_data_module(config, dict_train, dict_val, dict_test):
    if config["use_augmentation"] == True:
        transform_set = A.Compose([A.VerticalFlip(p=0.5),
                                A.HorizontalFlip(p=0.5),
                                A.RandomRotate90(p=0.5)])
    else:
        transform_set = None

    dm = OCS_DataModule(
        dict_train = dict_train,
        dict_val = dict_val,
        dict_test = dict_test,
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        drop_last = True,
        num_classes = config["num_classes"],
        num_channels = 5,
        use_metadata = config["use_metadata"],
        use_augmentations = transform_set)
    
    return dm

def get_segmentation_module(config):
    ## Define model and training parameters

    model = SMP_Unet_meta(n_channels = 5, 
                          n_classes = config["num_classes"], 
                          use_metadata = config["use_metadata"])
    
    if config["use_weights"] == True:
        with torch.no_grad():
            class_weights = torch.FloatTensor(config["class_weights"])
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

    seg_module = SegmentationTask(
        model = model,
        num_classes = config["num_classes"],
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        use_metadata=config["use_metadata"]
    )

    return seg_module


def train_model(config, data_module, seg_module):
        
    ## Define callbacks

    ckpt_callback = ModelCheckpoint(
        monitor = "val_loss",
        dirpath = os.path.join(out_dir,"checkpoints"),
        filename = "ckpt-{epoch:02d}-{val_loss:.2f}"+'_' + config['outputs']["out_model_name"],
        save_top_k = 1,
        mode = "min",
        save_weights_only = True, # can be changed accordingly
    )

    early_stop_callback = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.00,
        patience = 30, # if no improvement after 30 epoch, stop learning. 
        mode = "min",
    )

    prog_rate = TQDMProgressBar(refresh_rate=config["progress_rate"])

    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        prog_rate,
    ]

    logger = TensorBoardLogger(
        save_dir = out_dir,
        name = Path("tensorboard_logs"+'_'+config['outputs']["out_model_name"]).as_posix()
    )

    loggers = [
        logger
    ]

    ## Define trainer and run

    trainer = Trainer(
        accelerator = config["accelerator"],
        devices = config["gpus_per_node"],
        strategy = config["strategy"],
        num_nodes = config["num_nodes"],
        max_epochs = config["num_epochs"],
        num_sanity_val_steps = 0,
        callbacks = callbacks,
        logger = loggers,
        enable_progress_bar = config["enable_progress_bar"],
    )

    trainer.fit(seg_module, datamodule=data_module)

    ## Check metrics on validation set

    trainer.validate(seg_module, datamodule=data_module)


def predict(config, data_module, seg_module):

    ## Inference and predictions export

    writer_callback = PredictionWriter(        
        output_dir = os.path.join(out_dir, "predictions"+"_"+config['outputs']["out_model_name"]),
        write_interval = "batch",
    )

    #### instanciation of prediction Trainer
    trainer = Trainer(
        accelerator = config["accelerator"],
        devices = config["gpus_per_node"],
        strategy = config["strategy"],
        num_nodes = config["num_nodes"],
        callbacks = [writer_callback],
        enable_progress_bar = config["enable_progress_bar"],
    )

    trainer.predict(seg_module, datamodule=data_module, return_predictions=False)
    print('--  [FINISHED.]  --', f'output dir : {out_dir}', sep='\n')

if __name__ == "__main__":

    args = argParser.parse_args()
  
    config = read_config(args.config_file)

    out_dir = Path(config['outputs']["out_folder"], config['outputs']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(2022, workers=True)

    ## Define data sets
    dict_train, dict_val, dict_test = step_loading(config["data"], use_metadata=config["use_metadata"])  
    print_recap(config, dict_train, dict_val, dict_test)
    
    ## Define modules

    dm = get_data_module(config, dict_train, dict_val, dict_test)
    seg_module = get_segmentation_module(config)

    ## Train model

    train_model(config, dm, seg_module)

    ## Compute predictions

    predict(config, dm, seg_module)

    ## Compute mIoU over the predictions

    truth_msk = config['data']['path_labels_test']
    pred_msk  = os.path.join(out_dir, "predictions"+"_"+config['outputs']["out_model_name"])
    mIou, ious = generate_miou(truth_msk, pred_msk)
    print_metrics(mIou, ious)

    

    
