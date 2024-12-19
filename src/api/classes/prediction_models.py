from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class SupportedModel(Enum):
    Rgbi15clResnet34Unet = "rgbi-15cl-resnet34-unet"


class FlairModel(ABC):
    name: str
    bucket_name: str
    blob_prefix: str

    @property
    @abstractmethod
    def relative_weights_path(self) -> str:
        """This function returns the model weights path"""
        ...


@dataclass
class Rgbi15clResnet34UnetModel(FlairModel):
    name: str = "rgbi-15cl-resnet34-unet"
    bucket_name: str = "netcarbon-ign"
    blob_prefix: str = "FLAIR-1/model"

    @property
    def relative_weights_path(self) -> str:
        """This function returns the model weights path"""
        return "{}/FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth".format(
            self.blob_prefix
        )
