"""
Entry point of the Detect CLI tool.
This module aims to perform a detection based on an extent or a collection of extent
with an Odeon model and a dictionary of raster input
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, TypeAlias, Literal, List, Dict, Tuple, Any
import logging 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from jsonargparse import ArgumentParser  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
import geopandas as gpd  # noqa: E402
from shapely.geometry import box

from src.zone_detect.types import GEO_FLOAT_TUPLE, OUTPUT_TYPE # noqa: E402
from src.zone_detect.detect import Detector  # noqa: E402
from src.zone_detect.job import ZoneDetectionJob  # noqa: E402

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
    """Main entry point and configuration class for detection

    Implements
    ----------
    object
        base class

    Parameters
    ----------
    img_pixels_detection : int
        image size of output in pixel
    model_name : str
        name of te model as declared in the nn.models.build_model function
    model_weights : str
        file of trained model with weights parameters (ex: pytorch lightning checkpoint)
    n_classes : int
        The number of class learned by the model
    batch_size : int
        the size of the batch in the dataloader
    use_gpu : boolean
        use a GPU or not
    output_path : str
        output path of the detection
    output_type : str
        the output type, one of uint8, float32 or bit
    num_worker : int, optional
        Number of worker used by the dataloader.
        Be careful with a prediction by zone (concurrency), by default None (0 extra worker)

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
    output_type: OUTPUT_TYPE = "uint8"
    num_worker: Optional[int] = None
    _detector: Optional[Detector] = field(init=False)
    _df: pd.DataFrame | None = None


    def __post_init__(self):

        self.use_gpu = False if torch.cuda.is_available() is False else self.use_gpu
        self.df: pd.DataFrame | None = None

        STD_OUT_LOGGER.info(f"""
        ##############################################
        ZONE DETECTION 
        ##############################################\n 
        CUDA available? {torch.cuda.is_available()}\n
        |- output path : {self.output_path}
        
        |- input image path : {self.input_img_path}
        |- bands : {self.bands}
        |- noramlization : {self.normalization[0]['norm_type']}
        |- image size for detection: {self.img_pixels_detection}
        |- overlap margin: {self.margin}
        
        |- model weights path: {self.model_weights}
        |- model: {self.model_name}
        |- number of classes: {self.n_classes}
        |- output type: {self.output_type}
        |- device: {"cuda" if self.use_gpu else "cpu"}
        |- batch size: {self.batch_size}""")


        try:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            if not all(os.path.isfile(f) for f in [self.model_weights]):
                raise Exception("Model weight files do not exist.")
            self.configure()
        except Exception as error:
            LOGGER.debug(f"something went wrong during detection configuration {error}")



    def __call__(self):
        """Call the Detector implemented 
        """
        self.detector.run()



    def configure(self):
        """Configuraiton of the Detector class used to make the
        detection
        """

        input_image = self.input_img_path
        with rasterio.open(input_image) as src:
            crs = src.crs
            LOGGER.debug(crs)

            left, bottom, right, top = src.bounds
            gdf_zone = gpd.GeoDataFrame({"id": 1, 
                                         "out_name": input_image.split('/')[-1].split('.')[0]+'_pred',
                                         "left": left, "right": right,  
                                         "bottom": bottom, "top": top,                                         
                                         "geometry": [box(left, bottom, right, top)]},
                                         crs=crs,
                                         geometry="geometry",
                                         )
            
            self.resolution = round(src.res[0],5),  round(src.res[1],5)

        margin_zone = self.margin
        output_size = self.img_pixels_detection

        STD_OUT_LOGGER.info(f"""        |- resolution : {self.resolution}\n""")

        self.df = ZoneDetectionJob.build_job(gdf=gdf_zone,
                                             output_size=output_size,
                                             resolution=self.resolution,
                                             overlap=self.margin
                                             )
        LOGGER.debug(len(self.df))

        zone_detection_job = ZoneDetectionJob(self.df,
                                              self.output_path,
                                              )
        zone_detection_job.save_job()

        n_channel = len(self.bands)
        LOGGER.debug(f"number of channel input: {n_channel}")

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
