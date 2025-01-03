"""This module defines classes and enumerations for managing flair models"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class SupportedModel(Enum):
    """Models supported to run flair scripts"""

    Rgbi15clResnet34Unet = "rgbi-15cl-resnet34-unet"


class FlairModel(ABC):
    """Abstract base class representing a flair-detect model.

    Attributes:
        name (str): The name of the model.
        bucket_name (str): The name of the Google Cloud Storage bucket where
                           the model is stored.
        blob_prefix (str): The prefix path within the bucket where the model
                           files are located.

    Methods:
        relative_weights_path: Abstract property that should be implemented by
                              subclasses to return the model weights path.
    """

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
    """FlairModel for the Rgbi15clResnet34Unet model."""

    name: str = "rgbi-15cl-resnet34-unet"
    bucket_name: str = "netcarbon-ign"
    blob_prefix: str = "FLAIR-1/model"

    @property
    def relative_weights_path(self) -> str:
        """This function returns the model weights path"""
        return "{}/FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth".format(
            self.blob_prefix
        )
