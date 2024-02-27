"""
Entry point of the Detect CLI tool.
This module aims to perform a detection based on an extent or a collection of extent
with an Odeon model and a dictionary of raster input
"""

import os
from pathlib import Path
import warnings
from dataclasses import dataclass, field
from typing import Optional, List
import logging 
warnings.simplefilter(action='ignore', category=FutureWarning)

from jsonargparse import ArgumentParser  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
import geopandas as gpd  # noqa: E402
from shapely import wkt# noqa: E402
from shapely.geometry import box

from src.zone_detect.commons import Error, ErrorCodes  # noqa: E402
from src.zone_detect.commons import dirs_exist, files_exist, create_folder # noqa: E402
from src.zone_detect.detect import Detector  # noqa: E402
from src.zone_detect.job import ZoneDetectionJob  # noqa: E402
from src.zone_detect.types import GEO_FLOAT_TUPLE, OUTPUT_TYPE

LOGGER = logging.getLogger(__name__)
# test zone FID=18

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
    pixels_detection : int
        image size of output in pixel
    model_name : str
        name of te model as declared in the nn.models.build_model function
    checkpoint : str
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
    Error
        ERR_DETECTION_ERROR, if something goes wrong during the prediction
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

        STD_OUT_LOGGER.info(f"""
        ##############################################
        ZONE DETECTION 
        ##############################################\n 
        CUDA available? {torch.cuda.is_available()}""")

        self.use_gpu = False if torch.cuda.is_available() is False else self.use_gpu
        self.df: pd.DataFrame | None = None

        STD_OUT_LOGGER.info(f"""
        |- output path : {self.output_path}
        |- input image path : {self.input_img_path}
        |- model weights path: {self.model_weights}
        |- image size for detection: {self.img_pixels_detection}
        |- overlap margin: {self.margin}
        |- model: {self.model_name}
        |- number of classes: {self.n_classes}
        |- output type: {self.output_type}
        |- device: {"cuda" if self.use_gpu else "cpu"}
        |- batch size: {self.batch_size}""")

        try:
            create_folder(self.output_path)
            self.check()
            self.configure()
        except Error as error:
            raise error
        except Exception as error:
            raise Error(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def __call__(self):
        """Call the Detector implemented 
        """
        # LOGGER.debug(self.__dict__)
        self.detector.run()

    def check(self):
        """Check configuration
        if there is an anomaly in the input parameters.

        Raises
        ------
        Error
            ERR_DETECTION_ERROR, f something wrong has been detected in parameters
        """

        try:
            files_exist([self.model_weights])
            dirs_exist([self.output_path])
        except Error as error:
            raise Error(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)
        else:
            pass

    def configure(self):
        """Configuraiton of the Detector class used to make the
        detection
        """

        input_image = self.input_img_path
        with rasterio.open(input_image) as src:
            crs = src.crs
            LOGGER.debug(crs)

            left, bottom, right, top = src.bounds
            gdf_zone = gpd.GeoDataFrame({'id': 1, 
                                         'out_name': input_image.split('/')[-1].split('.')[0]+'_pred',
                                         'geometry': [box(left, bottom, right, top)]},
                                         crs=crs,
                                         geometry="geometry")
            
            self.resolution = round(src.res[0], 2), round(src.res[1], 2) 

        
        margin_zone = self.margin
        output_size = self.img_pixels_detection

        STD_OUT_LOGGER.info(f"""        |- resolution {self.resolution}\n""")

        self.df, _ = ZoneDetectionJob.build_job(gdf=gdf_zone,
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
    #print(f'configuration: {cfg}')
    runner = parser.instantiate_classes(cfg=cfg).conf
    runner()
