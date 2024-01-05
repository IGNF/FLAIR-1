import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A

from .model import smp_unet_mtd
from .data_module import flair_datamodule
from .task_module import segmentation_task_training, segmentation_task_predict



def get_data_module(config, dict_train=None, dict_val=None, dict_test=None):
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
        num_classes = config["num_classes"],
        use_metadata = config["use_metadata"],
        use_augmentations = transform_set
    )
    
    return dm



def get_segmentation_module(config, stage='train'):
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
    model = smp_unet_mtd(architecture = config['model_architecture'],
                         encoder = config['encoder_name'],
                         n_channels = 5, 
                         n_classes = config["num_classes"], 
                         use_metadata = config["use_metadata"]
    )
    
    if stage == 'train':
        if config["use_weights"]:
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

        seg_module = segmentation_task_training(
            model = model,
            num_classes = config["num_classes"],
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            use_metadata=config["use_metadata"]
        )

    elif stage == 'predict':
        seg_module = segmentation_task_predict(
            model = model,
            num_classes = config["num_classes"],
            use_metadata=config["use_metadata"],
        )        

    return seg_module