"""
module of Detection jobs
"""
import os
from pathlib import Path
import math
from logging import getLogger
import multiprocessing
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import rasterio
from rasterio.features import geometry_window
from rasterio.plot import reshape_as_raster

from src.zone_detect.dataset import ZoneDetectionDataset
from src.zone_detect.commons import Error, ErrorCodes
from src.zone_detect.rio import RIODatasetCollection
from src.zone_detect.image import TypeConverter, substract_margin
from src.zone_detect.commons import create_folder
from src.zone_detect.model import get_module, SegmentationModelFactory
from src.zone_detect.job import create_polygon_from_bounds, ZoneDetectionJob, ZoneDetectionJobNoDalle
from src.zone_detect.types import OUTPUT_TYPE, GEO_INT_TUPLE, GEO_FLOAT_TUPLE


LOGGER = getLogger(__name__)
NB_PROCESSOR = multiprocessing.cpu_count()


def load_model(checkpoint: str | Path, model_name: str = 'Unet',
               encoder: str = 'resnet34', n_classes: int = 15,
               n_channels: int = 5, *args, **kwargs) -> nn.Module:

    model: nn.Module = SegmentationModelFactory.create_model(model_name=model_name, encoder=encoder,
                                                                n_classes=n_classes, n_channels=n_channels)
    state_dict = get_module(checkpoint=checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=True)

    return model


class Detector:
    """
    """

    def __init__(self,
                 layers: str | Path,
                 bands: List,
                 norma: Dict,                 
                 margin_zone: int,
                 job: ZoneDetectionJob | ZoneDetectionJobNoDalle,
                 output_path: str | Path,
                 checkpoint: str | Path,
                 model_name: str = 'unet',
                 encoder_name: str = 'resnet34',
                 n_classes: int | None = 12,
                 n_channel: int = 3,
                 img_size_pixel: int | GEO_INT_TUPLE = 256,
                 resolution: GEO_FLOAT_TUPLE | None = None,
                 batch_size: int = 16,
                 use_gpu: bool = True,
                 num_worker: int | None = None,
                 output_type: OUTPUT_TYPE = "uint8",
                 out_dalle_size: int | None | bool = False
                 ):

        self.resolution: GEO_FLOAT_TUPLE = resolution if resolution is not None else [0.20, 0.20]
        self.img_size_pixel = img_size_pixel
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        if self.use_gpu:
            assert torch.cuda.is_available(), ('you chosse to infer on gpu but the command torch.cuda.is_available()'
                                               ' return False')
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.num_worker = num_worker
        self.output_path = output_path
        self.output_type = output_type
        self.job = job
        self.data_loader = None
        self.dataset = None
        self.model = load_model(checkpoint=self.checkpoint,
                                n_classes=self.n_classes, n_channels=self.n_channel, encoder=self.encoder_name,
                                model_name=self.model_name)
        self.model.eval()  # remove dropout and batchnorm for inference mode
        self.model.to(self.device)
        self.meta = None
        self.resolution = resolution
        self.gdal_options = {"compress": "LZW",
                             "tiled": True,
                             "blockxsize": self.img_size_pixel,
                             "blockysize": self.img_size_pixel,
                            }
        if self.output_type == "bit":
            self.gdal_options["bit"] = 1
        self.layers = layers
        self.norma = norma
        self.bands = bands
        self.margin_zone = margin_zone
        self.output_write = os.path.join(self.output_path, "result")
        create_folder(self.output_write)
        self.gdal_options["BIGTIFF"] = "YES"
        self.dst = None
        self.meta_output = None
        self.out_dalle_size = out_dalle_size
        self.rio_ds_collection = None
        self.converter = TypeConverter()
        LOGGER.debug(out_dalle_size)

    def configure(self):
        LOGGER.debug(len(self.job))
        self.rio_ds_collection = RIODatasetCollection()
        self.dst = rasterio.open(self.layers)
        self.meta = self.dst.meta.copy()
        self.meta["driver"] = "GTiff"
        self.meta["dtype"] = "uint8" if self.output_type in ["uint8", "bit", "argmax"] else "float32"
        self.meta["count"] = self.n_classes
        self.meta_output = self.meta.copy()
        if self.out_dalle_size is None or self.out_dalle_size is False:
            self.meta_output["height"] = self.img_size_pixel - (2 * self.margin_zone)
            self.meta_output["width"] = self.meta_output["height"]
        else:
            self.meta_output["height"] = math.ceil(self.out_dalle_size[1] / self.resolution[1])
            self.meta_output["width"] = math.ceil(self.out_dalle_size[0] / self.resolution[0])
        if self.output_type == "argmax":
            self.meta_output["count"] = 1
        self.num_worker = 0 if self.num_worker is None else self.num_worker
        torch.inference_mode()

    def detect(self, images):
        """
        Parameters
        ----------
        images

        Returns
        -------
        """

        if self.use_gpu:
            images = images.cuda()
        with torch.no_grad():
            logits = self.model(images)
            logits.to(self.device)
        # predictions
        predictions = torch.sigmoid(logits)
        predictions = predictions.cpu().numpy()
        return predictions

    def save(self, predictions, indices):
        for prediction, index in zip(predictions, indices):
            prediction = prediction.transpose((1, 2, 0)).copy()
            prediction = substract_margin(prediction, self.margin_zone, self.margin_zone)
            prediction = reshape_as_raster(prediction)

            prediction = self.converter.from_type("float32").to_type(self.output_type).convert(prediction)

            output_id = self.job.get_cell_at(index[0], "output_id")
            LOGGER.debug(output_id)
            name = str(output_id) + ".tif"
            output_file = os.path.join(self.output_write, name)

            if self.out_dalle_size is not None and self.rio_ds_collection.collection_has_key(output_id):
                out = self.rio_ds_collection.get_rio_dataset(output_id)
            else:
                left = self.job.get_cell_at(index[0], "left_o")
                bottom = self.job.get_cell_at(index[0], "bottom_o")
                right = self.job.get_cell_at(index[0], "right_o")
                top = self.job.get_cell_at(index[0], "top_o")
                if self.out_dalle_size is False:
                    self.meta_output["width"], self.meta_output["height"] = ((right - left) // self.resolution[0],
                                                                             (top - bottom) // self.resolution[1])
                self.meta_output["transform"] = rasterio.transform.from_bounds(
                    left, bottom, right, top, self.meta_output["width"], self.meta_output["height"])
                out = rasterio.open(output_file, 'w+', **self.meta_output, **self.gdal_options)
                LOGGER.debug(out.bounds)
                LOGGER.debug(self.dst.bounds)
                self.rio_ds_collection.add_rio_dataset(output_id, out)

            LOGGER.debug(out.meta)

            left = self.job.get_cell_at(index[0], "left")
            bottom = self.job.get_cell_at(index[0], "bottom")
            right = self.job.get_cell_at(index[0], "right")
            top = self.job.get_cell_at(index[0], "top")
            geometry = create_polygon_from_bounds(left, right, bottom, top)
            LOGGER.debug(geometry)
            window = geometry_window(out, [geometry], pixel_precision=6)
            window = window.round_shape(op='ceil', pixel_precision=4)
            LOGGER.debug(window)
            if self.output_type == "argmax":
                out.write(prediction, window=window)
            else:
                out.write_band([i for i in range(1, self.n_classes + 1)], prediction, window=window)
            self.job.set_cell_at(index[0], "job_done", 1)

            if self.out_dalle_size is not None and self.job.job_finished_for_output_id(output_id):

                self.rio_ds_collection.delete_key(output_id)
                self.job.mark_dalle_job_as_done(output_id)
                self.job.save_job()

            if self.out_dalle_size is None:

                out.close()


    def normalization(self, in_img, norma):
        if norma['norm_type'] not in ['scaling', 'custom']:
            print("Normalization argument should be 'scaling' or 'custom'")
            raise ValueError("Invalid normalization type")

        if norma['norm_type'] == 'custom':
            if len(norma['norm_means']) != len(norma['norm_stds']):
                print("If custom, provided normalization means and stds should be of the same length.")
                raise ValueError("Mismatched means and stds lengths")
            else:
                for i in range(len(norma['norm_means'])):
                    in_img[:, i] -= norma['norm_means'][i]/255
                    in_img[:, i] /= norma['norm_stds'][i]/255
        elif norma['norm_type'] == 'scaling':
            pass
        
        return in_img

  
    def run(self):
        if len(self.job) > 0:
            try:
                self.dataset = ZoneDetectionDataset(job=self.job,
                                                    layers=self.layers,
                                                    bands=self.bands,
                                                    output_type=self.output_type,
                                                    meta=self.meta,
                                                    height=self.img_size_pixel,
                                                    width=self.img_size_pixel,
                                                    resolution=self.resolution)
                with self.dataset as dataset:
                    LOGGER.debug(f"length: {len(dataset.job)}")
                    self.data_loader = DataLoader(dataset,
                                                  batch_size=min(self.batch_size, len(self.job)),
                                                  num_workers=self.num_worker,
                                                  pin_memory=True
                                                  )
                    for samples in tqdm(self.data_loader):
                        imgs = samples["image"]
                        imgs = self.normalization(imgs, self.norma)
                        predictions = self.detect(imgs)
                        indices = samples["index"].cpu().numpy()
                        LOGGER.debug(indices)
                        self.save(predictions, indices)
            except KeyboardInterrupt as error:
                LOGGER.warning("the job has been prematurely interrupted")
                raise Error(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)
            except rasterio._err.CPLE_BaseError as error:
                LOGGER.warning(f"CPLE error {error}")
            except Exception as error:
                raise Error(ErrorCodes.ERR_DETECTION_ERROR,
                                 "something went wrong during detection",
                                 stack_trace=error)
            finally:
                self.job.save_job()
                if self.dst is not None:
                    self.dst.close()
        else:
            LOGGER.warning(f""""job has no work to do, maybe
            your input directory or csv file is empty, or you may have set the
            interruption_recovery at true while the output directory
            {self.output_path} contain a job file of previous work completed
            (all the work has been done)""")
