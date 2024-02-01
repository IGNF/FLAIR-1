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
from logging import getLogger  # noqa: E402
warnings.simplefilter(action='ignore', category=FutureWarning)

from jsonargparse import ArgumentParser, Namespace  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
import geopandas as gpd  # noqa: E402
from shapely import wkt  # noqa: E402

from src.detect.commons import Error, ErrorCodes  # noqa: E402
from src.detect.commons import dirs_exist, files_exist, Zone  # noqa: E402
from src.detect.logger import get_new_logger, get_simple_handler  # noqa: E402
from src.detect.rio import get_number_of_band  # noqa: E402
from src.detect.detect import Detector  # noqa: E402
from src.detect.job import ZoneDetectionJob, ZoneDetectionJobNoDalle, create_box_from_bounds  # noqa: E402
from src.detect.types import GEO_FLOAT_TUPLE, OUTPUT_TYPE

LOGGER = getLogger(__name__)
# test zone FID=18

" A logger for big message "
STD_OUT_LOGGER = get_new_logger("stdout_detection")
ch = get_simple_handler()
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
    verbosity : boolean
        verbosity of logger
    img_size_pixel : int
        image size of output in pixel
    resolution : Union(float, tuple(float, float))
        output resolution in x and y
    model_name : str
        name of te model as declared in the nn.models.build_model function
    checkpoint : str
        file of trained model with weights parameters (ex: pytorch lightning checkpoint)
    zone : Zone
        the description of an inference by zone, by default None
        See Also :ref:`Zone <zone>`
    n_classes : int
        The number of class learned by the model
    batch_size : int
        the size of the batch in the dataloader
    use_gpu : boolean
        use a GPU or not
    interruption_recovery : boolean
        store and restart from where the detection has been
        if an interruption has been encountered
    mutual_exclusion : boolean
        In multiclass model you can use softmax if True or
        Sigmoïd if False
    output_path : str
        output path of the detection
    model_prefix: str, optional, default None
    key_path: List[str], optional, default None
        list of keywords to find the model in a checkpoint like pytorch lightning
        checkpoints, where the model is serializer with other elements
    output_type : str
        the output type, one of uint8, float32 or bit
    sparse_mode : boolean
        if set to True, will only write the annotated pixels on disk.
        If can save a lot of space if you pick up "bit" as output type
    threshold : float between 0 and 1
        threshold used in the case of an output in bit (0/1)
    num_worker : int, optional
        Number of worker used by the dataloader.
        Be careful with a prediction by zone (concurrency), by default None (0 extra worker)
    num_thread : int, optional
        Number of thread used during the prediction.
        Useful when you infer on CPU, by default None

    Raises
    ------
    Error
        ERR_DETECTION_ERROR, if something goes wrong during the prediction
    """
    verbosity: bool
    img_size_pixel: int
    resolution: float | GEO_FLOAT_TUPLE
    checkpoint: str
    zone: Zone
    n_classes: int
    batch_size: int
    use_gpu: bool
    output_path: str | Path
    model_prefix: str
    key_path: List[str] | None = None
    n_channels: int = 3
    interruption_recovery: bool = False
    mutual_exclusion: bool = True
    model_name: str = 'unet'
    encoder_name: str = 'resnet34'
    output_type: OUTPUT_TYPE = "uint8"
    sparse_mode: bool = False
    threshold: float = 0.5
    num_worker: Optional[int] = None
    num_thread: Optional[int] = None
    _detector: Optional[Detector] = field(init=False)
    _df: pd.DataFrame | None = None

    def __post_init__(self):
        self.resolution = self.resolution if isinstance(self.resolution,tuple) else (self.resolution, self.resolution)
        STD_OUT_LOGGER.info(f"""CUDA available? {torch.cuda.is_available()}""")
        self.use_gpu = False if torch.cuda.is_available() is False else self.use_gpu
        self.df: pd.DataFrame | None = None

        STD_OUT_LOGGER.info(f"""
        device: {"cuda" if self.use_gpu else "cpu"}
        model: {self.model_name}
        checkpoint: {self.checkpoint}
        number of classes: {self.n_classes}
        batch size: {self.batch_size}
        image size pixel: {self.img_size_pixel}
        resolution: {self.resolution}
        activation: {"softmax" if self.mutual_exclusion is True else "sigmoïd"}
        output type: {self.output_type}""")
        STD_OUT_LOGGER.info(f"""overlap margin: {self.zone.margin}
                            compute digital elevation model: {self.zone.dem}
                            tile factor: {self.zone.tile_factor}
                            """)

        try:
            self.check()
            self.configure()
        except Error as error:
            raise error
        except Exception as error:
            raise Error(ErrorCodes.ERR_DETECTION_ERROR,
                             "something went wrong during detection configuration",
                             stack_trace=error)

    def __call__(self):
        """Call the Detector implemented (by zone, or by dataset)
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
            files_exist([self.checkpoint])
            dirs_exist([self.output_path])
            if self.output_type == "argmax":
                assert self.mutual_exclusion, ('argmax output has been thought '
                                               'to work with mutual exclusion (softmax activation function')
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
        layers = self.zone.layers
        with rasterio.open(next(iter(layers)).path) as src:
            crs = src.crs
            bounds = src.bounds
            LOGGER.debug(crs)
        if os.path.isfile(self.zone.extent):
            gdf_zone = gpd.GeoDataFrame.from_file(self.zone.extent)
        elif isinstance(self.zone.extent, str):
            gdf_zone = gpd.GeoDataFrame([{"id": 1, "geometry": wkt.loads(self.zone.extent)}],
                                        geometry="geometry",
                                        crs=crs)
        elif self.zone.extent is None and len(self.zone.layers) == 1:
            gdf_zone = gpd.GeoDataFrame({"id": 1, "geometry": create_box_from_bounds(x_min=bounds[0],
                                                                                     y_min=bounds[1],
                                                                                     x_max=bounds[2],
                                                                                     y_max=bounds[3])},
                                        geometry="geometry",
                                        crs=crs)
        else:
            raise Error(ErrorCodes.ERR_MAIN_CONF_ERROR, message='Please provide one layer or an extent'
                                                                'with one or more layers')
        LOGGER.debug(gdf_zone)
        # extent = self.zone.extent
        tile_factor = self.zone.tile_factor
        margin_zone = self.zone.margin
        output_size = self.img_size_pixel * tile_factor
        out_dalle_size = self.zone.output_dalle_size if self.zone.output_dalle_size is not None else None
        LOGGER.debug(f"output_size {out_dalle_size}")
        self.df, _ = ZoneDetectionJob.build_job(gdf=gdf_zone,
                                                output_size=output_size,
                                                resolution=self.resolution,
                                                overlap=self.zone.margin,
                                                out_dalle_size=out_dalle_size)
        LOGGER.debug(len(self.df))
        # self.df = self.df.sample(n=100, random_state=1).reset_index()
        # self.df.to_file("/home/dlsupport/data/33/ground_truth/2018/learning_zones/test_zone_1.shp")

        if out_dalle_size is not None:
            zone_detection_job = ZoneDetectionJob(self.df,
                                                  self.output_path,
                                                  self.interruption_recovery)
        else:
            zone_detection_job = ZoneDetectionJobNoDalle(self.df,
                                                         self.output_path,
                                                         self.interruption_recovery)
        zone_detection_job.save_job()
        # write_job = WriteJob(df_write, self.output_path, self.interruption_recovery, file_name="write_job.shp")
        # write_job.save_job()
        dem = self.zone.dem
        n_channel = get_number_of_band(layers, dem)
        LOGGER.debug(f"number of channel input: {n_channel}")

        self.detector = Detector(layers=layers,
                                 tile_factor=tile_factor,
                                 margin_zone=margin_zone,
                                 job=zone_detection_job,
                                 output_path=self.output_path,
                                 model_name=self.model_name,
                                 checkpoint=self.checkpoint,
                                 encoder_name=self.encoder_name,
                                 n_classes=self.n_classes,
                                 n_channel=n_channel,
                                 model_prefix=self.model_prefix,
                                 key_path=self.key_path,
                                 img_size_pixel=self.img_size_pixel,
                                 resolution=self.resolution,
                                 batch_size=self.batch_size,
                                 use_gpu=self.use_gpu,
                                 num_worker=self.num_worker,
                                 num_thread=self.num_thread,
                                 mutual_exclusion=self.mutual_exclusion,
                                 output_type=self.output_type,
                                 sparse_mode=self.sparse_mode,
                                 threshold=self.threshold,
                                 verbosity=self.verbosity,
                                 dem=dem,
                                 out_dalle_size=out_dalle_size)

        LOGGER.debug(self.detector.__dict__)
        self.detector.configure()


def main():

    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_dataclass_arguments(theclass=ConfDetection, nested_key='conf', help=ConfDetection.__doc__)
    cfg = parser.parse_args()
    print(f'configuration: {cfg}')
    runner = parser.instantiate_classes(cfg=cfg).conf
    runner()
