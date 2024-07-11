#general 
import datetime
import os
import argparse 
import shutil
import torch
import sys

from pathlib import Path
from datetime import timedelta

from pytorch_lightning import seed_everything 
from pytorch_lightning.utilities.rank_zero import rank_zero_only  

from src.flair.tasks import train, predict
from src.flair.tasks_utils import get_data_module, get_segmentation_module, gather_paths
from src.flair.metrics import metrics
from src.flair.utils import read_config, print_recap



argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="Path to the .yaml config file", required=True)


def setup_environment(args):
    """
    This function reads the configuration file, creates the output directory, 
    and sets up the logger.
    """
    config = read_config(args.conf)
    out_dir = Path(config['paths']["out_folder"], config['paths']["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    #setup_logger(config)
    return config, out_dir 

@rank_zero_only
class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8') 
        self.encoding = self.terminal.encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def get_datasets(config):
    """
    Get the datasets for training, validation, and testing.
    """
    dict_train, dict_val, dict_test = None, None, None
    if config['tasks']['train']:
        dict_train = gather_paths(config, split='train')
        dict_val   = gather_paths(config, split='val')
    if config['tasks']['predict']: 
        dict_test  = gather_paths(config, split='test')
    return dict_train, dict_val, dict_test

@rank_zero_only
def copy_csv_and_config(config, out_dir, args):
    """
    Copy the CSV files and configuration file to the output directory.
    """
    csv_copy_dir = Path(out_dir, 'used_csv_and_config')
    csv_copy_dir.mkdir(parents=True, exist_ok=True)
    if config["tasks"]["train"]:
        shutil.copy(config["paths"]["train_csv"], csv_copy_dir)
        shutil.copy(config["paths"]["val_csv"], csv_copy_dir)
    if config["tasks"]["predict"]: shutil.copy(config["paths"]["test_csv"], csv_copy_dir)
    shutil.copy(args.conf, csv_copy_dir)


def load_checkpoint(ckpt_file_path, seg_module, exit_on_fail=False):
    """
    Load model weights from a checkpoint file.

    Parameters:
    ckpt_file_path (str): Path to the checkpoint file.
    seg_module: Segmentation module for training or prediction.
    exit_on_fail (bool): Whether to raise a SystemExit if the checkpoint file is invalid.
    """
    print('---------------------------------------------------------------') 
    if ckpt_file_path is not None and os.path.isfile(ckpt_file_path) and ckpt_file_path.endswith('.ckpt'):
        checkpoint = torch.load(ckpt_file_path, map_location="cpu")
        seg_module.load_state_dict(checkpoint["state_dict"], strict=False)
        print('--------------- Loaded model weights from ckpt. ---------------')
    elif ckpt_file_path is not None and os.path.isfile(ckpt_file_path) and (ckpt_file_path.endswith('.pth') or ckpt_file_path.endswith('.pt')):
        checkpoint = torch.load(ckpt_file_path, map_location="cpu")
        seg_module.load_state_dict(checkpoint, strict=False)
        print('----------- Loaded model weights from pytorch file. -----------')       
    else: 
        print("Invalid checkpoint file path.")
        if exit_on_fail:
            raise SystemExit()
    print('---------------------------------------------------------------')


def training_stage(config, data_module, out_dir):
    """
    Conducts the training stage of the model: sets up the training environment, loads the model weights from a checkpoint if available,
    trains the model, and logs the training information.

    Parameters:
    config (dict): Configuration dictionary containing parameters for the task.
    data_module: Data module for training, validation, and testing.
    seg_module: Segmentation module for training.
    out_dir (Path): Path object representing the output directory.

    Returns:
    trained_state_dict (OrderedDict): The state dictionary of the trained model.
    """
    start = datetime.datetime.now()

    seed_everything(config['seed'], workers=True)

    seg_module = get_segmentation_module(config, stage='train')

    if config['tasks']['train_tasks']['init_weights_only_from_ckpt']:
        load_checkpoint(config['paths']['ckpt_model_path'], seg_module, exit_on_fail=False)

    ckpt_callback = train(config, data_module, seg_module, out_dir)

    best_trained_state_dict = torch.load(ckpt_callback.best_model_path, map_location=torch.device('cpu'))['state_dict']

    end = datetime.datetime.now()
    inference_time_seconds = end - start
    inference_time_seconds = inference_time_seconds.total_seconds()

    print(f"\n[Training finished in {str(timedelta(seconds=inference_time_seconds))} HH:MM:SS with {config['num_nodes']} nodes and {config['gpus_per_node']} gpus per node]") 
    print(f"Model path : {os.path.join(out_dir,'checkpoints')}\n\n")
    print('\n'+'-'*40)

    return best_trained_state_dict
   
def predict_stage(config, data_module, out_dir_predict, trained_state_dict=None):
    """
    Conducts the prediction stage of the model: sets up the prediction environment, loads the model weights from the training stage or a checkpoint file,
    and makes predictions.

    Parameters:
    config (dict): Configuration dictionary containing parameters for the task.
    data_module: Data module for training, validation, and testing.
    out_dir_predict (Path): Path object representing the output directory for predictions.
    trained_state_dict (OrderedDict, optional): The state dictionary of the trained model. Defaults to None.
    """
    seg_module = get_segmentation_module(config, stage='predict')
    if config['tasks']['train']:
        seg_module.load_state_dict(trained_state_dict, strict=False)  
    else:
        load_checkpoint(config['paths']['ckpt_model_path'], seg_module)
    predict(config, data_module, seg_module, out_dir_predict)


def main():
    # Read config and create output folder
    args = argParser.parse_args()
    config, out_dir = setup_environment(args)

    # Custom Logger for console/logfile output
    sys.stdout = Logger(
        Path(config['paths']["out_folder"], config['paths']["out_model_name"], 'flair-compute.log').as_posix())
    print(datetime.datetime.now().strftime("Starting : %Y-%m-%d  %H:%M") + '\n')

    # Define data sets
    dict_train, dict_val, dict_test = get_datasets(config)
    print_recap(config, dict_train, dict_val, dict_test)

    # Copy relevant files for tracking
    if config["cp_csv_and_conf_to_output"]:
        copy_csv_and_config(config, out_dir, args)

    # Get LightningDataModule
    dm = get_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)

    # Initialize variable for weights
    trained_state_dict = None

    # Training
    if config['tasks']['train']:
        trained_state_dict = training_stage(config, dm, out_dir)

    # Inference
    if config['tasks']['predict']:
        out_dir_predict = Path(out_dir, 'predictions_' + config['paths']["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)

        # Metrics
        if config['tasks']['metrics']:
            metrics(config, out_dir_predict, remove_preds=config['tasks']['delete_preds'])


if __name__ == "__main__":
    main()
