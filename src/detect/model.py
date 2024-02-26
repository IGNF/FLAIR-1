import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Mapping
from logging import getLogger

from segmentation_models_pytorch import create_model
import torch


logger = getLogger(__name__)


def get_module(checkpoint: str | Path) -> Mapping:
    
    if checkpoint is not None and os.path.isfile(checkpoint) and checkpoint.endswith('.ckpt'):
        ckpt = torch.load(checkpoint, map_location='cpu')
        d = ckpt['state_dict']
    elif checkpoint is not None and os.path.isfile(checkpoint) and (checkpoint.endswith('.pth') or checkpoint.endswith('.pt')):
        d = torch.load(checkpoint, map_location='cpu')
    else:
        logger.error('Error with checkpoint provided : either a .ckpt with a "state_dict" key or a OrderedDict pt/pth file') 

    out = d    
        
    if 'model.seg_model' in list(out.keys())[0]:
        out = {k.partition('model.seg_model.')[2]: v for k, v in out.items()}
        out = {k: v for k, v in out.items() if k != ""}
        
    return out


@dataclass
class SegmentationModelFactory:
    model_name: str = 'unet'
    encoder: str = 'resnet34'
    n_channels: int = 5
    n_classes: int = 15
    _model: torch.nn.Module | None = field(init=False)

    def __post_init__(self):
        self._model = create_model(arch=self.model_name, encoder_name=self.encoder,
                                   classes=self.n_classes, in_channels=self.n_channels)

    def get(self):
        return self._model

    @classmethod
    def create_model(cls, model_name: str = 'unet', encoder: str = 'resnet34', n_channels: int = 5,
                     n_classes: int = 15):
        return SegmentationModelFactory(model_name=model_name,
                                        encoder=encoder, n_channels=n_channels, n_classes=n_classes).get()
