from pathlib import Path
from typing import Dict, Callable, Optional, List
from logging import getLogger

import torch
import rasterio
# from rasterio.plot import reshape_as_raster
import numpy as np

from src.detect.image import CollectionDatasetReader
from src.detect.commons import create_folder, Layer
from src.detect.job import ZoneDetectionJob, ZoneDetectionJobNoDalle
from src.detect.types import GEO_FLOAT_TUPLE, OUTPUT_TYPE, PARAMS

LOGGER = getLogger(__name__)


class ToDoubleTensor:
    """Convert ndarrays of sample(image, mask) into Tensors"""

    def __call__(self, **sample):
        image, mask = sample['image'], sample['mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        mask = mask.transpose((2, 0, 1)).copy()
        return {
            'image': torch.as_tensor(image, dtype=torch.float),
            'mask': torch.as_tensor(mask, dtype=torch.float)
        }


class ToSingleTensor:
    """Convert ndarrays of image into Tensors"""

    def __call__(self, **sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        return torch.from_numpy(image).float()


class ToPatchTensor:
    """Convert ndarrays of image, scalar index, affineasndarray into Tensors"""

    def __call__(self, **sample):

        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        index = sample["index"]
        affine = sample["affine"]
        return {
                "image": torch.from_numpy(image).float(),
                "index": torch.from_numpy(index).int(),
                "affine": torch.from_numpy(affine).float()
                }


class ToWindowTensor:
    """Convert ndarrays of image, scalar index"""

    def __call__(self, **sample):

        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()
        index = sample["index"]

        return {
                "image": torch.from_numpy(image).float(),
                "index": torch.from_numpy(index).int()
                }


class Compose:
    """Compose function differs from torchvision Compose as sample argument is passed unpacked to match albumentation
    behaviour.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **sample):
        for t in self.transforms:
            sample = t(**sample)
        return sample


class ZoneDetectionDataset:
    def __init__(self,
                 job: ZoneDetectionJobNoDalle | ZoneDetectionJob,
                 resolution: GEO_FLOAT_TUPLE,
                 width: int,
                 height: int,
                 layers: List[Layer],
                 output_type: OUTPUT_TYPE,
                 meta: Dict[str, PARAMS],
                 transform: Optional[Callable] = None,
                 dem: bool = False,
                 gdal_options: Dict[str, PARAMS] = None,
                 export_input: bool = False,
                 export_path: Optional[str | Path] = None
                 ):

        self.job: ZoneDetectionJobNoDalle | ZoneDetectionJob = job
        self.job.keep_only_todo_list()
        self.width: int = width
        self.height: int = height
        self.resolution: GEO_FLOAT_TUPLE = resolution
        self.transform_function: Callable = transform
        self.layers: List[Layer] = layers
        self.gdal_options = gdal_options
        self.dem = dem
        self.export_input = export_input
        self.export_path = export_path
        self.meta = meta
        self.to_tensor = ToWindowTensor()
        if self.export_path is not None:
            create_folder(self.export_path)

    def __enter__(self):
        for layer in self.layers:
            if self.gdal_options is None:
                layer.connexion = rasterio.open(layer.path)
            else:
                layer.connexion = rasterio.open(layer.path, **self.gdal_options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer in self.layers:
            layer.connexion.close()

    def __len__(self):
        return len(self.job)

    def __getitem__(self, index):
        try:
            bounds = self.job.get_bounds_at(index)
            # LOGGER.info(image_file)
            # LOGGER.info(self.image_files)
            img = CollectionDatasetReader.get_stacked_window_collection(self.layers,
                                                                        bounds,
                                                                        self.width,
                                                                        self.height,
                                                                        self.resolution,
                                                                        self.dem)
            # affine = meta["transform"]
            # LOGGER.debug(affine)
            sample = {"image": img, "index": np.asarray([index])}
            sample = self.to_tensor(**sample)
            return sample

        except rasterio._err.CPLE_BaseError as error:
            LOGGER.warning(f"CPLE error {error}")
