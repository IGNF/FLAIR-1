from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Mapping
from logging import getLogger

from segmentation_models_pytorch import create_model
import torch


logger = getLogger(__name__)

def get_module(checkpoint: str | Path, key_path: List[str] | None = None,
               model_prefix: str | None = None) -> Mapping:
    d = torch.load(checkpoint, map_location='cpu')

    print(key_path)
    out = d
    if key_path is not None:
        for i in key_path:
            try:
                out = out[i]
            except KeyError as ke:
                logger.error(f'Key {i} from key_path parameter was not found in checkpoint')

    # assert isinstance(out, dict)

    if model_prefix is not None:
        out = {k.partition('model.seg_model.')[2]: v for k, v in out.items()}
        out = {k: v for k, v in out.items() if k != ""}

    return out


@dataclass
class SegmentationModelFactory:
    model_name: str = 'unet'
    encoder: str = 'resnet34'
    n_channels: int = 5
    n_classes: int = 12
    _model: torch.nn.Module | None = field(init=False)

    def __post_init__(self):
        self._model = create_model(arch=self.model_name, encoder_name=self.encoder,
                                   classes=self.n_classes, in_channels=self.n_channels)

    def get(self):
        return self._model

    @classmethod
    def create_model(cls, model_name: str = 'unet', encoder: str = 'resnet34', n_channels: int = 5,
                     n_classes: int = 12):
        return SegmentationModelFactory(model_name=model_name,
                                        encoder=encoder, n_channels=n_channels, n_classes=n_classes).get()
