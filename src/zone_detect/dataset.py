import torch
import rasterio
import numpy as np

from pathlib import Path
from logging import getLogger
from typing import Dict, Callable, Optional, List, Tuple, Any

import rasterio.transform
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
from skimage.util import img_as_float

from src.zone_detect.job import ZoneDetectionJob

LOGGER = getLogger(__name__)


class ToWindowTensor:
    """
    Convert ndarrays of image and scalar index.

    This class transforms input samples by transposing image dimensions
    and converting them to PyTorch tensors.
    """

    def __call__(self, **sample):
        image = sample['image']
        image = image.transpose((2, 0, 1)).copy()
        scalar_index = sample["index"]

        return {
            "image": torch.from_numpy(image).float(),
            "index": torch.from_numpy(scalar_index).int()
        }


class ZoneDetectionDataset:
    def __init__(self,
                 job: ZoneDetectionJob,
                 resolution: Tuple[float, float],
                 width: int,
                 height: int,
                 layers: str | Path,
                 bands: List,
                 meta: Dict[str, Dict[str, Any]],
                 transform: Optional[Callable] = None,
                 export_input: bool = False,
                 export_path: Optional[str | Path] = None
                 ):

        self.job: ZoneDetectionJob = job
        self.job.keep_only_todo_list()
        self.width: int = width
        self.height: int = height
        self.resolution: Tuple[float, float] = resolution
        self.transform_function: Callable = transform
        self.layers: str | Path = layers
        self.bands: List = bands
        self.export_input = export_input
        self.export_path = export_path
        self.meta = meta
        self.to_tensor = ToWindowTensor()

    def __enter__(self):
        self.layers = rasterio.open(self.layers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.layers.close()

    def __len__(self):
        return len(self.job)

    def __getitem__(self, index):
        try:
            bounds = self.job.get_bounds_at(index)
            src = self.layers
            window = rasterio.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.meta["transform"])

            img = src.read(indexes=self.bands, window=window, out_shape=(len(self.bands), self.height, self.width),
                           resampling=Resampling.bilinear, boundless=True,
                  )
            
            img = reshape_as_image(img)

            if img.ndim == 2:
                img = img[..., np.newaxis]

            img = img_as_float(img)  

            LOGGER.debug(f"type of img: {type(img)}, shape {img.shape}")

            sample = {"image": img, "index": np.asarray([index])}
            sample = self.to_tensor(**sample)

            return sample

        except rasterio._err.CPLE_BaseError as error:
            LOGGER.warning(f"CPLE error {error}")
