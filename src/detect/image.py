from pathlib import Path
from typing import Any, Dict
from logging import getLogger

import numpy as np
from math import isclose
from typing import List, Optional, cast, Tuple
from rasterio.enums import Resampling
import rasterio.transform
import rasterio.windows
from rasterio.windows import Window
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.util import img_as_float

from src.detect.rio import get_bounds, create_patch_from_center
from src.detect.types import GEO_FLOAT_TUPLE, OUTPUT_TYPE
from src.detect.commons import Layer

LOGGER = getLogger(__name__)


def raster_to_ndarray_from_dataset(
        src: rasterio.io.DatasetReader,
        width: int,
        height: int,
        resolution: Optional[GEO_FLOAT_TUPLE] = None,
        band_indices: Optional[List[int]] = None,
        resampling: Resampling = Resampling.bilinear,
        window: Window = None,
        boundless: bool = True):
    """Load and transform an image into a ndarray according to parameters:
    - center-cropping to fit width, height
    - loading of specific bands to fit image_bands array

    Parameters
    ----------
    src : rasterio.DatasetReader
        raster source for the conversion
    width : int
        output image width
    height : int
        output image height
    resolution: obj:`list` of :obj: `number`
        output resolution for x and y
    band_indices : obj:`list` of :obj: `int`, optional
        list of band indices to be loaded in output image, by default None (native image bands are used)
    resampling: one enum from rasterio.Resampling
        resampling method to use when a resolution change is necessary.
        Default: Resampling.bilinear
    window: rasterio.window, see rasterio docs
        use a window in rasterio format or not to select a subsection of the raster
        Default: None
    Returns
    -------
    out: Tuple[ndarray, dict]
        a numpy array representing the image, and the metadata in rasterio format.
    """
    " get the width and height at the target resolution "
    if band_indices is None:
        band_indices = range(1, src.count + 1)
    if window is None and resolution is None:
        raise ValueError("windows and resolution can not be set to None")
    if window is None:
        left, bottom, right, top = src.bounds

        def get_dim_bounds(dim_size: int,
                           dim_res: float,
                           dim_min: float,
                           dim_max: float) -> Tuple[float, float]:
            dim_dist = dim_size * dim_res
            dim_close = isclose(dim_dist, dim_max-dim_min, rel_tol=1e-04)
            if dim_close:
                return dim_min, dim_max
            dim_center = (dim_max + dim_min) / 2.0
            if dim_dist < dim_max-dim_min:
                # crop image from center
                out_max = dim_center + dim_dist
                out_min = dim_center - dim_dist
            else:
                # not enought data raise error
                msg = f"could get not get out res = {dim_res} and out size = {dim_size}"
                msg += f"from coord bound = [{dim_min}, {dim_max}]"
                raise ValueError(msg)
            return out_min, out_max
        left, right = get_dim_bounds(width, resolution[0], left, right)
        bottom, top = get_dim_bounds(height, resolution[1], bottom, top)
        window = rasterio.windows.from_bounds(left, bottom, right, top, src.meta["transform"])
    else:
        left, bottom, right, top = rasterio.windows.bounds(window, src.meta["transform"])
    img = src.read(
        indexes=band_indices, window=window, out_shape=(len(band_indices), height, width),
        resampling=resampling, boundless=boundless)
    " reshape img from gdal band format to numpy ndarray format "
    img = reshape_as_image(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    meta = src.meta.copy()
    LOGGER.debug(meta)
    affine = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    meta["transform"] = affine
    return img, meta


def raster_to_ndarray(image_file: str | Path,
                      width: int,
                      height: int,
                      resolution: GEO_FLOAT_TUPLE,
                      band_indices=None,
                      resampling: Any =Resampling.bilinear,
                      window=None) -> Tuple[np.ndarray, Dict]:
    """Simple helper function to call raster_to_ndarray_from_dataset from
    a raster path file contrary to a rasterio.Dataset

    Parameters
    ----------
    image_file : str or Path
        raster source for the conversion
    width : int, optional
        output image width, by default None (native image width is used)
    height : int, optional
        output image height, by default None (native image height is used)
    resolution: obj:`tuple` of :obj: `float`
        output resolution for x and y
    band_indices : obj:`list` of :obj: `int`, optional
        list of band indices to be loaded in output image, by default None (native image bands are used)
    resampling: one enum from rasterio.Resampling
        resampling method to use when a resolution change is necessary.
        Default: Resampling.bilinear
    window: rasterio.window, see rasterio docs
        use a window in rasterio format or not to select a subsection of the raster
        Default: None
    Returns
    -------
    out: Tuple[ndarray, dict]
        a numpy array representing the image, and the metadata in rasterio format.

    """
    LOGGER.debug(image_file)
    with rasterio.open(image_file) as src:
        if resolution is None:
            resolution = src.res
        if width is None:
            width = src.width
        if height is None:
            height = src.height
        return raster_to_ndarray_from_dataset(src, width, height, resolution, band_indices, resampling, window)


def crop_center(img, cropx, cropy):
    """Crop numpy array based on the center of
    array (rounded to inf element for array of even size)
    We expect array to be of format W*H*C
    Parameters
    ----------
    img : numpy NDArray
        numpy array of dimension 2 or 3
    cropx : int
        size of crop on x axis (first axis)
    cropy : int
        size of crop on x axis (second axis)

    Returns
    -------
    numpy NDArray
        the cropped numpy 3d array
    """
    if img.ndim == 2:
        y, x = img.shape
    else:
        y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def substract_margin(img, margin_x, margin_y):
    """Crop numpy array based on the center of
    array (rounded to inf element for array of even size)
    We expect array to be of format H*W*C
    Parameters
    ----------
    img : numpy NDArray
        numpy array of dimension 2 or 3
    margin_x : int
        size of crop on x axis (first axis)
    margin_y : int
        size of crop on x axis (second axis)

    Returns
    -------
    numpy NDArray
        the substracted of its margin numpy 3darray
    """
    if img.ndim == 2:

        y, x = img.shape

    else:

        y, x, _ = img.shape

    return img[0 + margin_y: y - margin_y, 0 + margin_x:x - margin_x]


class TypeConverter:
    """Simple class to handle conversion of output format
    in detection notably.
    """
    def __init__(self):

        self._from = "float32"
        self._to = "uint8"

    def from_type(self, img_type):
        """get orignal type

        Parameters
        ----------
        img_type : str
            we actually handle only the case of 32 bits

        Returns
        -------
        TypeConverter
            itself
        """

        self._from = img_type
        return self

    def to_type(self, img_type: OUTPUT_TYPE):
        """get target type

        Parameters
        ----------
        img_type : str
            we actually handle float32, int8, and bit

        Returns
        -------
        TypeConverter
            self
        """

        self._to = img_type
        return self

    def convert(self, img, threshold=0.5):
        """Make conversion

        Parameters
        ----------
        img : NDArray
            input image
        threshold : float, optional
            used with 1bit output, to binarize pixels, by default 0.5

        Returns
        -------
        NDArray
            converted image
        """

        if self._from == "float32":
            if self._to == "float32":
                return img
            elif self._to == "uint8":
                if img.max() > 1:
                    info = np.iinfo(img.dtype)  # Get the information of the incoming image type
                    img = img.astype(np.float32) / info.max  # normalize the data to 0 - 1
                img = np.iinfo(np.uint8).max * img  # scale by 255
                return img.astype(np.uint8)
            elif self._to == "bit":
                img = img > threshold
                return img.astype(np.uint8)
            elif self._to == "argmax":
                img = np.argmax(img, axis=0)
                return np.expand_dims(img.astype(np.uint8), axis=0)
            else:
                LOGGER.warning("the output type has not been interpreted")
                return img


def ndarray_to_affine(affine):
    """
    Parameters
    ----------
    affine : numpy Array

    Returns
    -------
    rasterio.Affine
    """
    return rasterio.Affine(affine[0], affine[1], affine[2], affine[3], affine[4], affine[5])


def compute_dem(dsm_img: np.ndarray, dtm_img: np.ndarray) -> np.ndarray:
    # raw dem = dsm - dtm
    img = dsm_img - dtm_img
    # LOGGER.debug(img.sum())
    # dsm should not be under dtm theorically but this could happen
    # due to product specification (rounding) etc.. so add check
    mne_msk = dtm_img > dsm_img
    img[mne_msk] = 0
    # scaling to vertical resolution such that it should be ok when convert
    # to uint8
    img *= 5  # empircally chosen factor

    # probably a wrong thing to do.. resolution[0] and 1 are x and y resolution
    # and not min/max elevation
    # xmin, xmax = max(resolution[0], resolution[1]), 255
    # img[img < xmin] = xmin  # low pass filter
    # img[img > xmax] = xmax  # high pass filter

    # low pass filer, should not do anatyhing as we should
    # already have only positives values.
    img[img < 0] = 0
    # high pass filter.
    img[img > 255] = 255
    return img


class CollectionDatasetReader:
    """Static class to handle connection fo multiple raster input

    Returns
    -------
    NDArray
        stacked bands of multiple rasters
    """
    @staticmethod
    def get_stacked_window_collection(layers: List[Layer],
                                      bounds: List[int],
                                      width: int,
                                      height: int,
                                      resolution: GEO_FLOAT_TUPLE,
                                      dem: bool = False,
                                      resampling: Resampling = Resampling.bilinear):
        """Stack multiple raster band in one raster, with a specific output format and resolution
        and output bounds. It can handle the DEM = DSM - DTM computation if the dict of raster
        band includes a band called "DTM" and a band called "DSM", but you must set the dem parameter
        to True.

        Parameters
        ----------
        layers : list of Layer
            a list of raster layer definition with the nomenclature
             Layer contains "path": "/path/to/my/raster", "band": an array of index band of interest
             in rasterio/gdal format (starting to 1)}
        bounds : Union[Tuple, List]
            bounds delimiting the output extent
        width : int
            the width of the output
        height : int
            the height of the output
        resolution: obj:`tuple` of :obj: `float`
            the output resolution for x and y
        dem : bool, optional
            indicate if a DSM - DTM band is computed on the fly
            ("DSM" and "DTM" must be present in the dictionary of raster), by default False
        resampling: one enum from rasterio.Reampling
            resampling method to use when a resolution change is necessary.
            Default: Resampling.bilinear

        Returns
        -------
        numpy NDArray
            the stacked raster
        """
        handle_dem = False
        stacked_bands = None
        keys = list()
        has_dsm: bool = False
        has_dtm: bool = False
        dsm_index: Optional[int] = None
        dtm_index: Optional[int] = None

        for idx, layer in enumerate(layers):
            key = layer.key
            keys.append(key)
            if key == "DSM":
                has_dsm = True
                dsm_index = idx
            if key == "DTM":
                has_dtm = True
                dtm_index = idx
        if has_dsm in keys and has_dtm in keys and dem is True:
            handle_dem = True

        for layer in layers:
            if (layer.key not in ["DSM", "DTM"]) or handle_dem is False:
                src = layer.connexion
                window = rasterio.windows.from_bounds(
                    bounds[0], bounds[1], bounds[2], bounds[3], src.meta["transform"])
                band_indices = layer.bands
                img, _ = raster_to_ndarray_from_dataset(src,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=band_indices,
                                                        resampling=resampling,
                                                        window=window)

                # pixels are normalized to [0, 1]
                img = img_as_float(img)

                LOGGER.debug(f"type of img: {type(img)}, shape {img.shape}")

                stacked_bands = img if stacked_bands is None else np.dstack([stacked_bands, img])
                LOGGER.debug(f"type of stacked bands: {type(stacked_bands)}, shape {stacked_bands.shape}")

        if handle_dem:
            dsm_ds = layers[dsm_index].connexion
            dtm_ds = layers[dtm_index].connexion
            dsm_window = rasterio.windows.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], dsm_ds.meta["transform"])
            dtm_window = rasterio.windows.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], dtm_ds.meta["transform"])
            dsm_band_indices = layers[dsm_index].bands
            dtm_band_indices = layers[dtm_index].bands
            assert len(dsm_band_indices) == len(dtm_band_indices) == 1
            dsm_img, _ = raster_to_ndarray_from_dataset(dsm_ds,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=dsm_band_indices,
                                                        resampling=resampling,
                                                        window=dsm_window)

            dtm_img, _ = raster_to_ndarray_from_dataset(dtm_ds,
                                                        width,
                                                        height,
                                                        resolution,
                                                        band_indices=dtm_band_indices,
                                                        resampling=resampling,
                                                        window=dtm_window)

            # normalize to [0, 1]. as input are float img_as_float could not be used for
            # normalization.
            img = compute_dem(dsm_img=dsm_img, dtm_img=dtm_img)
            img = img / 255
            # LOGGER.debug(f"img min: {img.min()}, img max: {img.max()}, img shape: {img.shape}")
            stacked_bands = img if stacked_bands is None else np.dstack([stacked_bands, img])
            LOGGER.debug(f"type of stacked bands: {type(stacked_bands)}, shape {stacked_bands.shape}")

        return stacked_bands


# TODO, delete stack_window_raster function
    @staticmethod
    def stack_window_raster(center,
                            layers: List[Layer],
                            meta,
                            dem,
                            compute_only_masks=False,
                            raster_out=None,
                            meta_msk=None,
                            output_type="uint8"):
        """stack band at window level of geotif layer and create a
        couple patch image and patch mask

        Parameters
        ----------
        center : pandas.core.series.Series
            a row from a pandas DataFrame
        layers : list of Layer
            dictionary of layer geo tif (RGB, CIR, etc.)
        meta : dict
            metadata for the window raster
        dem : boolean
            rather calculate or not DMS - DMT and create a new band with it
        compute_only_masks : int (0,1)
            rather compute only masks or not
        raster_out: str
            path of rasterized full mask where to extract the window mask
        meta_msk: dict
            metadata in rasterio format for raster mask
        output_type: str
            output type of output patch raster in numpy dtype format

        Returns
        -------
            None
        """
        resampling = Resampling.bilinear

        bounds = get_bounds(center.x,
                            center.y,
                            meta['width'],
                            meta['height'],
                            meta['resolution'][0],
                            meta['resolution'][1])

        window = rasterio.windows.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3], meta['transform'])

        # Adapt meta for the patch
        meta_img_patch = meta.copy()
        meta_img_patch['transform'] = rasterio.windows.transform(window, meta['transform'])

        meta_msk_patch = meta_msk.copy()
        meta_msk_patch["transform"] = rasterio.windows.transform(window, meta_msk['transform'])

        # Creation of masks tiles from an input window.
        create_patch_from_center(center["msk_file"],
                                 raster_out,
                                 meta_msk_patch,
                                 window,
                                 resampling)

        if not compute_only_masks:
            img = CollectionDatasetReader.get_stacked_window_collection(layers=layers,
                                                                        bounds=bounds,
                                                                        width=meta_img_patch['width'],
                                                                        height=meta_img_patch['height'],
                                                                        resolution=meta_img_patch['resolution'],
                                                                        dem=dem,
                                                                        resampling=resampling)
            raster_img = (reshape_as_raster(img) * 255).astype(output_type)

            with rasterio.open(center["img_file"], 'w', **meta_img_patch) as dst:
                dst.write(raster_img)