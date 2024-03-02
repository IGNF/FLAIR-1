import os
from pathlib import Path
from typing import Optional, List, Literal
from dataclasses import dataclass, field
import logging 
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import rasterio
import torch
import geopandas as gpd
from shapely.geometry import box
from jsonargparse import ArgumentParser

from src.zone_detect.detect import Detector  
from src.zone_detect.job import ZoneDetectionJob


#### LOGGERS 
LOGGER = logging.getLogger(__name__)

log = logging.getLogger('stdout_detection')
log.setLevel(logging.DEBUG)
STD_OUT_LOGGER = log

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
STD_OUT_LOGGER.addHandler(ch)



@dataclass
class ConfDetection:

    """
    Main entry point and configuration class for detection.

    Parameters
    ----------
    output_path : str | Path
        Output path for the detection results.
    input_img_path : str | Path
        Path to the input image.
    bands : List
        List of bands (related to image data).
    model_weights : str
        File path to the trained model with weight parameters (e.g., PyTorch Lightning checkpoint).
    normalization : List
        List of normalization-related information.
    n_classes : int
        Number of classes learned by the model.
    batch_size : int
        Batch size for the dataloader.
    use_gpu : bool
        Whether to use a GPU for computation.
    img_pixels_detection : int, optional
        Image size of the output in pixels (default: 512).
    margin : int, optional
        Margin value (default: 256).
    model_name : str, optional
        Name of the model (as declared in nn.models.build_model function, default: 'unet').
    encoder_name : str, optional
        Name of the encoder (default: 'resnet34').
    output_type : Literal['class_prob', 'argmax'], optional
        Output type (default: 'class_prob').
    num_worker : Optional[int], optional
        Number of workers used by the dataloader (default: None).

    Raises
    ------
    Exception
    """

    output_path: str | Path
    input_img_path: str | Path
    bands: List
    model_weights: str
    normalization: List
    n_classes: int
    batch_size: int
    use_gpu: bool
    img_pixels_detection: int = 512
    margin: int = 256
    model_name: str = 'unet'
    encoder_name: str = 'resnet34'
    output_type: Literal['class_prob', 'argmax'] = "class_prob"
    num_worker: Optional[int] = None
    _detector: Optional[Detector] = field(init=False)
    _df: pd.DataFrame | None = None


    def log_detection_configuration(self):
        """
        Log detection configuration details.
        """
        STD_OUT_LOGGER.info(f"""
        ##############################################
        ZONE DETECTION
        ##############################################

        CUDA available? {torch.cuda.is_available()}

        |- output path: {self.output_path}
        |- output raster name: {self.output_name + '.tif'}

        |- input image path: {self.input_img_path}
        |- bands: {self.bands}
        |- resolution: {self.resolution}

        |- image size for detection: {self.img_pixels_detection}
        |- overlap margin: {self.margin}
        |- number of classes: {self.n_classes}
        |- normalization: {self.normalization[0]['norm_type']}
        |- output type: {self.output_type}
        
        |- model weights path: {self.model_weights}
        |- model: {self.model_name}
        |- device: {"cuda" if self.use_gpu else "cpu"}
        |- batch size: {self.batch_size}


        """)


    def __post_init__(self):

        self.use_gpu = (False if torch.cuda.is_available() is False else self.use_gpu)
        self.df: pd.DataFrame | None = None

        try:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if not all(os.path.isfile(f) for f in [self.model_weights]):
                raise Exception("Model weight files do not exist.")
            self.configure()
            self.log_detection_configuration()
        except Exception as error:
            LOGGER.debug(f"Something went wrong during detection configuration: {error}")


    def __call__(self):
        """
        Call the Detector implemented 
        """
        self.detector.run()



    def configure(self):
        """
        Configuration of the Detector class used for detection.
        """
        input_image = self.input_img_path
        self.output_name = '_'.join([input_image.split('/')[-1].split('.')[0],
                                self.output_type,
                                self.normalization[0]['norm_type'],
                                'pred'])

        with rasterio.open(input_image) as src:
            crs = src.crs
            LOGGER.debug(crs)

            left, bottom, right, top = src.bounds
            bounding_box_gdf = gpd.GeoDataFrame(
                {
                    "id": 1,
                    "out_name": self.output_name,
                    "left": left,
                    "right": right,
                    "bottom": bottom,
                    "top": top,
                    "geometry": [box(left, bottom, right, top)],
                },
                crs=crs,
                geometry="geometry",
            )

            self.resolution = round(src.res[0], 5), round(src.res[1], 5)

        margin_zone = self.margin
        self.output_size = self.img_pixels_detection

        self.df = ZoneDetectionJob.build_job(
            gdf=bounding_box_gdf,
            patch_size=self.output_size,
            resolution=self.resolution,
            margin=self.margin,
        )

        zone_detection_job = ZoneDetectionJob(self.df, self.output_path, file_name=self.output_name+'_slicing.gpkg')
        zone_detection_job.save_job()

        n_channel = len(self.bands)
        LOGGER.debug(f"Channels: {n_channel}")

        self.detector = Detector(layers=input_image,
                                 bands=self.bands,
                                 norma=self.normalization[0],
                                 margin_zone=margin_zone,
                                 job=zone_detection_job,
                                 output_path=self.output_path,
                                 model_name=self.model_name,
                                 checkpoint=self.model_weights,
                                 encoder_name=self.encoder_name,
                                 n_classes=self.n_classes,
                                 n_channel=n_channel,
                                 img_size_pixel=self.img_pixels_detection,
                                 resolution=self.resolution,
                                 batch_size=self.batch_size,
                                 use_gpu=self.use_gpu,
                                 num_worker=self.num_worker,
                                 output_type=self.output_type,
                                 )

        LOGGER.debug(self.detector.__dict__)
        self.detector.configure()
  

def main():

    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_dataclass_arguments(theclass=ConfDetection, nested_key='conf', help=ConfDetection.__doc__)
    cfg = parser.parse_args()
    runner = parser.instantiate_classes(cfg=cfg).conf
    runner()
