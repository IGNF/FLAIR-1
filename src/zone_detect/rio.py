from typing import List

import numpy as np
import rasterio
from rasterio import features
from logging import getLogger


LOGGER = getLogger(__name__)

IMAGE_TYPE = {
                "uint8": [0, 0, 2**8 - 1, np.uint8, rasterio.uint8],
                "uint16": [1, 0, 2**16 - 1, np.uint16, rasterio.uint16]
}



def rasterize_shape(tuples, meta, shape, fill=0, default_value=1):
    """

    Parameters
    ----------
    tuples :list[tuples]
    meta : dict
    shape
    fill : int
    default_value

    Returns
    -------

    """
    raster = features.rasterize(tuples,
                                out_shape=shape,
                                default_value=default_value,
                                transform=meta["transform"],
                                dtype=rasterio.uint8,
                                fill=fill)
    return raster


def get_window_param(center, dataset, width, height):
    """
    get window left right top bottom to exctract path from the Raster

    Parameters
    ----------
    center : pandas.core.series.Series
     a row from a pandas DataFrame
    dataset : rasterio.DatasetReader
     a Rasterio Dataset to get the row, col from x, y in GeoCoordinate
     this is where we will extract path
    width : float
     width in Geocoordinate
    height : float
     height GeoCoordinate

    Returns
    -------
    Tuple
     col, row, width, height

    """

    row, col = dataset.index(center.x, center.y)
    col_s = int(height / 2)
    row_s = int(width / 2)
    return col - col_s, row - row_s, width, height


def get_bounds(x, y, width, height, resolution_x, resolution_y):
    """
    get window left right top bottom to exctract path from the Raster

    Parameters
    ----------
    x : float
     x coord
    y: float
     y coord
    width : float
     width in Geocoordinate
    height : float
     height GeoCoordinate
    resolution_x: float
     resolution in x axis
    resolution_y: float
     resolution in y axis

    Returns
    -------
    Tuple
     left, bottom, right, top

    """

    x_side = 0.5 * width * resolution_x
    y_side = 0.5 * height * resolution_y
    left, top, right, bottom = x - x_side, y + y_side, x + x_side, y - y_side
    return left, bottom, right, top


def create_patch_from_center(out_file, msk_raster, meta, window, resampling):
    """Create mask

    Parameters
    ----------
    out_file: str
    msk_raster : tif of raster
    meta : dict
     geo metadata in gdal format
    window : rasterio.window.Window
     rasterio window
    resampling : rasterio.enums.Resampling
     resampling method (billinear, cubic, etc.)

    Returns
    -------

    """

    with rasterio.open(msk_raster) as dst:

        clip = dst.read(window=window, out_shape=(meta["count"], meta["height"], meta["width"]), resampling=resampling)
        bands = clip[0:clip.shape[0]-1].astype(np.bool).astype(np.uint8).copy()
        other_band = np.sum(bands, axis=0, dtype=np.uint8)
        other_band = (other_band == 0).astype(np.uint8)

        with rasterio.open(out_file, 'w', **meta) as raster_out:
            out = dst.read(window=window,
                           out_shape=(meta["count"], meta["height"], meta["width"]),
                           resampling=resampling)
            out = np.vstack((out[0:out.shape[0]-1], np.array([other_band])))
            raster_out.write(out)

        return window


def check_proj(dict_of_raster):
    """

    Parameters
    ----------
    dict_of_raster : dict
     dictionary of raster name, raster file path and band list

    Returns
    -------
    boolean
     True if all rasters have same crs
    """

    check = True
    crs_compare = None

    for raster_name, raster in dict_of_raster.items():

        with rasterio.open(raster) as src:

            crs = src.meta["crs"]
            crs_compare = crs if crs_compare is None else crs_compare

            if crs_compare != crs:

                check = False

    return check


def count_band_for_stacking(dict_of_raster):
    """
    take a dictionnary of raster (name of raster, path_to_file)
    and return the number of band necessary to stack them in a single raster
    Parameters
    ----------
    dict_of_raster: dict

    Returns
    -------
    int
    """

    nb_of_necessary_band = 0
    rasters: dict = dict_of_raster.copy()

    if ("DSM" and "DTM") in rasters:

        nb_of_necessary_band += 1

    rasters.pop("DSM", None)
    rasters.pop("DTM", None)

    for raster_name, raster in rasters.items():

        with rasterio.open(raster) as src:

            count = src.meta["count"]
            nb_of_necessary_band += count

    return nb_of_necessary_band


def normalize_array_in(array, dtype, max_type_val):
    """ Normalize band based on the encoding type and the max value of the type
    example: to convert in uint16 type will be uint16 and max type value will be 65535

    Parameters
    ----------
    array : NdArray
     band to normalize
    dtype : Union[str, numpy.dtype, rastion.dtype]
     target data type
    max_type_val : Union[int, float]
     value

    Returns
    -------
    band : NdArray
     the band normalized in the targeted encoding type

    """

    array = array.astype(np.float64)
    LOGGER.debug(array.max())
    LOGGER.debug(f"type {dtype}")

    if float(array.max()) != float(0):

        array *= max_type_val / array.max()

    return array.astype(dtype)


def get_max_type(rasters):
    """Find the type of the patches generated

    Parameters
    ----------
    rasters : dict
     a dictionary of raster name with at least metadata path for each one

    Returns
    -------
    dtype : str
     encoding type for the patches
    """

    dtype = "uint8"

    for name, raster in rasters.items():

        for r in raster["path"]:

            with rasterio.open(r) as src:

                LOGGER.debug(f"raster: {raster}, type: {src.meta['dtype']}")
                if src.meta["dtype"] in IMAGE_TYPE.keys() and IMAGE_TYPE[src.meta["dtype"]][0] > IMAGE_TYPE[dtype][0]:

                    dtype = src.meta["dtype"]

    LOGGER.debug(f"dtype: {dtype}")
    return dtype


def affine_to_ndarray(affine):
    """
    Convert an affine transform into an ndarray
    Used for pytorch embedding

    Parameters
    ----------
    affine : rasterio.Affine

    Returns
    -------
    numpy NdArray
    """

    tuple_affine = (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)
    return np.asarray(tuple_affine)


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



def affine_to_tuple(affine):
    """
    Convert an affine transform into a tuple
    Used for pytorch embedding

    Parameters
    ----------
    affine : rasterio.Affine

    Returns
    -------
    list
    """

    return (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)


class RIODatasetCollection:

    def __init__(self):

        self.collection = {}

    def add_rio_dataset(self, key, rio_ds):

        self.collection[key] = rio_ds

    def get_rio_dataset(self, key):

        if self.collection_has_key(key):

            return self.collection[key]

        else:

            return None

    def collection_has_key(self, key):

        return key in self.collection

    def delete_key(self, key):

        if self.collection_has_key(key):

            self.collection[key].close()
            del self.collection[key]