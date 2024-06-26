import os
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer
from src.flair.writer import predictionwriter

def train(config, data_module, seg_module, out_dir):
    """
    Trains a model using the provided data module and segmentation module.
    
    Parameters:
    config (dict): Configuration dictionary containing parameters for training.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training.
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
    if config['tasks']['continue_training']:
        print("-------------CONTINUE TRAINING----------------")
        print('----------------------------------------------')
        trainer.fit(seg_module, datamodule=data_module, ckpt_path=config['paths']['ckpt_model_path'])
    else:
        trainer.fit(seg_module, datamodule=data_module)

    ## Check metrics on validation set
    trainer.validate(seg_module, datamodule=data_module)




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