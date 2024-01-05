#general 
import os
import argparse 
import logging

from pathlib import Path
from datetime import timedelta

#deep learning
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything 


#flair-one baseline modules 
from src.model_utils import get_data_module, get_segmentation_module
from src.utils import gather_paths, read_config, print_recap_training, print_metrics, setup_logger
from src.writer import predictionwriter
from src.metrics import generate_miou


argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")



def train_model(config, data_module, seg_module, out_dir):
    """
    This function trains a model using the provided data module and segmentation module.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for training.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training or prediction.

    Returns:
    None
    """

    ## Define callbacks
    ckpt_callback = ModelCheckpoint(
        monitor = "val_loss",
        dirpath = os.path.join(out_dir,"checkpoints"),
        filename = "ckpt-{epoch:02d}-{val_loss:.2f}"+'_' + config['paths']["out_model_name"],
        save_top_k = 1,
        mode = "min",
        save_weights_only = False, # can be changed accordingly
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
    trainer.fit(seg_module, datamodule=data_module)

    ## Check metrics on validation set
    trainer.validate(seg_module, datamodule=data_module)




def predict(config, data_module, seg_module, out_dir):
    """
    This function makes predictions using the provided data module and segmentation module.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for prediction.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training or prediction.
    out_dir (str): Output directory for saving the predictions.

    Returns:
    None
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

    
    

if __name__ == "__main__":

    # Read config and create output folder
    args = argParser.parse_args()
    config = read_config(args.config_file)
    out_dir = Path(config['paths']["out_folder"], config['paths']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seting up de logger
    setup_logger(config)
    
    # Recording training time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   
    starter.record() 

    # Define data sets
    dict_train, dict_val, dict_test = None, None, None
    if config['tasks']['train']:
        dict_train = gather_paths(config, split='train')
        dict_val   = gather_paths(config, split='val')
    if config['tasks']['predict']: 
        dict_test  = gather_paths(config, split='test')

    dm = get_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)

    if config['tasks']['train']:

        # Set the seed
        seed_everything(config['seed'], workers=True)

        print_recap_training(config, dict_train, dict_val)
        
        # Define modules
        seg_module = get_segmentation_module(config, stage='train')

        # Train model
        train_model(config, dm, seg_module, out_dir)
        trained_state_dict = seg_module.state_dict()

        ender.record()  
        inference_time_seconds = starter.elapsed_time(ender) / 1000.0   

        logging.info(f"\n[Training finished in {str(timedelta(seconds=inference_time_seconds))} HH:MM:SS with {config['num_nodes']} nodes and {config['gpus_per_node']} gpus per node]") 
        logging.info(f"Model path : {os.path.join(out_dir,'checkpoints')}\n\n")
        logging.info('\n'+'-'*40)


    if config['tasks']['predict']:

        seg_module = get_segmentation_module(config, stage='predict')

        if config['tasks']['train']:
            out_dir = Path(out_dir, 'predictions_'+config['paths']["out_model_name"])
            out_dir.mkdir(parents=True, exist_ok=True)
            seg_module.load_state_dict(trained_state_dict, strict=False)
            
        else:
            out_dir = Path(config['paths']["out_folder"], 'predictions')
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_file_path = config['paths']['ckpt_model_path']
            if ckpt_file_path is not None and os.path.isfile(ckpt_file_path) and ckpt_file_path.endswith('.ckpt'):
                # Load the checkpoint
                checkpoint = torch.load(ckpt_file_path, map_location="cpu")
                seg_module.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                logging.error("Invalid checkpoint file path. A valid .ckpt file is mandatory if training is disabled!")
                raise SystemExit()

        # Infer model
        predict(config, dm, seg_module, out_dir)

        logging.info(f'\n[Inference finished of {len(dict_test["IMG"])} files]\n'+f'output dir : {out_dir}')


    if config['tasks']['metrics']:

        truth_msk = sorted(dict_test['MSK'], key=lambda x: int(x.split('_')[-1][:-4]))
        pred_msk  = os.path.join(out_dir)
        mIou, ious = generate_miou(truth_msk, pred_msk)
        print_metrics(mIou, ious)





     




