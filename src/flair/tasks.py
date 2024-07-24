import os
from pathlib import Path
import torch
import torch.nn as nn
import sys


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer
from src.flair.writer import predictionwriter


def check_batchnorm_and_batch_size(config, seg_module):
    """
    Check if the model contains BatchNorm layers and if the batch size is 1.
    If both conditions are met, print a message and abort the script.

    Parameters:
    config (dict): Configuration dictionary containing parameters for training.
    seg_module (nn.Module): Segmentation module for training.
    """
    batch_size = config['batch_size']  # Assuming batch_size is in config

    for module in seg_module.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and batch_size == 1:
            print("Warning: The model contains BatchNorm layers and the batch size is set to 1.")
            print("Aborting training to avoid potential issues.")
            print("PLease set a batch size >1 in the current configuration.")
            sys.exit(1)  # Exit the script



def train(config, data_module, seg_module, out_dir):
    """
    Trains a model using the provided data module and segmentation module.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for training.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training.
    """

    check_batchnorm_and_batch_size(config, seg_module)

    ## Define callbacks
    ckpt_callback = ModelCheckpoint(
        monitor = config['ckpt_monitor'],
        dirpath = os.path.join(out_dir,"checkpoints"),
        filename = "ckpt-{epoch:02d}-{val_loss:.2f}"+'_' + config['paths']["out_model_name"],
        save_top_k = 1,
        mode = config['ckpt_monitor_mode'],
        save_last=config['ckpt_save_also_last'],
        verbose=config['ckpt_verbose'],
        save_weights_only = config['ckpt_weights_only'], 
    )

    early_stop_callback = EarlyStopping(
        monitor = config['ckpt_monitor'],
        min_delta = 0.00,
        patience = config['ckpt_earlystopping_patience'], 
        mode = config['ckpt_monitor_mode'],
    )

    prog_rate = TQDMProgressBar(refresh_rate=config["progress_rate"])

    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        prog_rate,
    ]

    logger = TensorBoardLogger(
        save_dir = out_dir,
        name = Path("tensorboard_logs"+'_'+config['paths']["out_model_name"]).as_posix()
    )

    loggers = [
        logger
    ]

    ## Define trainer
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

    ## Train model
    if config['tasks']['train_tasks']['resume_training_from_ckpt']:
        print('---------------------------------------------------------------')                      
        print('------------- RESUMING TRAINING FROM CKPT_PATH ----------------')
        print('---------------------------------------------------------------')
        checkpoint = torch.load(config['paths']['ckpt_model_path'])
        trainer.fit(seg_module, datamodule=data_module, ckpt_path=config['paths']['ckpt_model_path'])
              
    else:    
        trainer.fit(seg_module, datamodule=data_module)

    ## Check metrics on validation set
    trainer.validate(seg_module, datamodule=data_module)
    
    return ckpt_callback



def predict(config, data_module, seg_module, out_dir):
    """
    This function makes predictions using the provided data module and segmentation module.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for prediction.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for prediction.
    out_dir (str): Output directory for saving the predictions.
    """
    ## Inference and predictions export

    writer_callback = predictionwriter(   
        config,
        output_dir = os.path.join(out_dir),
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
