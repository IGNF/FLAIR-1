import torch
import torch.nn as nn
import os
from pathlib import Path
from logging import getLogger
from dataclasses import dataclass, field
from typing import Mapping

import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoConfig



LOGGER = getLogger(__name__)

@dataclass
class FLAIR_ModelFactory:
    """
    A factory class for creating models based on the provided configuration.
    This class supports models from both SegmentationModelsPytorch and HuggingFace. 
    """

    config: Mapping
    model_provider: str = field(init=False)
    seg_model: nn.Module = field(init=False)

    def __post_init__(self):

        self.model_provider = self.config['model_framework']['model_provider']

        n_channels = int(len(self.config['channels']))
        n_classes = self.config["n_classes"]

        if self.model_provider == 'SegmentationModelsPytorch':
            encoder, architecture = self.config['model_framework']['SegmentationModelsPytorch']['encoder_decoder'].split('_')
            self.seg_model = smp.create_model(
                arch=architecture,
                encoder_name=encoder,
                classes=n_classes,
                in_channels=n_channels
            )

        elif self.model_provider == 'HuggingFace':
            cfg_model = AutoConfig.from_pretrained(
                self.config['model_framework']['HuggingFace']['org_model'], 
                num_labels=n_classes
            )
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                self.config['model_framework']['HuggingFace']['org_model'], 
                config=cfg_model, 
                ignore_mismatched_sizes=True
            )

    def forward(self, x, met=None):
        if self.model_provider == 'SegmentationModelsPytorch':
            output = self.seg_model(x)
        elif self.model_provider == 'HuggingFace':
            output = self.seg_model(x)
        return output



def get_module(checkpoint: str | Path) -> Mapping:
    if checkpoint is not None and os.path.isfile(checkpoint):
        weights = torch.load(checkpoint, map_location='cpu')
        if checkpoint.endswith('.ckpt'):
            weights = weights['state_dict']
    else:
        LOGGER.error('Error with checkpoint provided: either a .ckpt with a "state_dict" key or an OrderedDict pt/pth file')
        return {}

    if 'model.seg_model' in list(weights.keys())[0]:
        weights = {k.partition('model.seg_model.')[2]: v for k, v in weights.items()} 
        weights = {k: v for k, v in weights.items() if k != ""}

    return weights



def load_model(config: Mapping) -> nn.Module:
    checkpoint = config.get('model_weights')

    model_factory = FLAIR_ModelFactory(config)
    model = model_factory.seg_model

    state_dict = get_module(checkpoint=checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=True)

    return model
