import os
import torch
import torch.nn as nn

from pathlib import Path
from segmentation_models_pytorch import create_model
from dataclasses import dataclass, field
from typing import Mapping
from logging import getLogger

LOGGER = getLogger(__name__)

@dataclass
class SegmentationModelFactory:
    """
    Factory for creating segmentation models.
    """

    model_name: str = 'unet'
    encoder: str = 'resnet34'
    n_channels: int = 5
    n_classes: int = 19
    segmentation_model: torch.nn.Module | None = field(init=False)

    def __post_init__(self):
        self.segmentation_model = create_model(
            arch=self.model_name,
            encoder_name=self.encoder,
            classes=self.n_classes,
            in_channels=self.n_channels
        )

    def get(self) -> torch.nn.Module:
        return self.segmentation_model

    @classmethod
    def create_model(cls, model_name: str = 'unet', encoder: str = 'resnet34', n_channels: int = 5,
                     n_classes: int = 15) -> torch.nn.Module:
        return SegmentationModelFactory(
            model_name=model_name,
            encoder=encoder,
            n_channels=n_channels,
            n_classes=n_classes
        ).get()
    
    

def get_module(checkpoint: str | Path) -> Mapping:
    if checkpoint is not None and os.path.isfile(checkpoint):
        weights = torch.load(checkpoint, map_location='cpu')
        if checkpoint.endswith('.ckpt'):
            weights = weights['state_dict']
    else:
        LOGGER.error('Error with checkpoint provided: either a .ckpt with a "state_dict" key or an OrderedDict pt/pth file')

    if 'model.seg_model' in list(weights.keys())[0]:
        weights = {k.partition('model.seg_model.')[2]: v for k, v in weights.items()} 
        weights = {k: v for k, v in weights.items() if k != ""}

    return weights


    
def load_model(checkpoint: str | Path, model_name: str = 'Unet',
               encoder: str = 'resnet34', n_classes: int = 15,
               n_channels: int = 5, *args, **kwargs) -> nn.Module:

    model: nn.Module = SegmentationModelFactory.create_model(model_name=model_name, encoder=encoder,
                                                                n_classes=n_classes, n_channels=n_channels)
    state_dict = get_module(checkpoint=checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=True)

    return model