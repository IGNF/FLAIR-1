import os
import pathlib
from pathlib import Path
from enum import Enum, unique, auto
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, List
from rasterio.io import DatasetReader


@dataclass
class Layer:
    """Represent a source of raster data
    Parameters
    ----------

    path: str | Path
    bands: Optional[list[int]], default: None
    name: str, default: 'image'
    key: str = 'img'
    connection: Optional[rasterio.io.DatasetReader]
    """
    path: str | Path
    bands: Optional[list[int]] = None
    name: str = 'image'
    key: str = 'image'
    connection: Optional[DatasetReader] = None


@dataclass
class Zone:
    """
    Represent the description of an inference by zone, by default None
    Implements
    ----------
    object
        base class

    Parameters
    ----------
    layers: Dict[str, Layer]
    extent: str | Path
    margin: int | None, default: None
    output_dalle_size: int | None, default: None
    dem: bool, default: False
    tile_factor: int | None, default: None
    """
    layers: List[Layer]
    extent: str | Path | None = None
    margin: int | None = None
    output_dalle_size: int | None = None
    dem: bool = False
    tile_factor: int | None = None


class Error(Exception):
    """
    Custom Exception for Odeon project
    """

    def __init__(self, error_code, message='', stack_trace=None, *args, **kwargs):
        """

        Parameters
        ----------
        error_code : ErrorCodes
         code of the error
        message : str
        stack_trace : object
         trace of python Exception possibly with this error
        args
        kwargs
        """
        # Raise a separate exception in case the error code passed isn't specified in the ErrorCodes enum
        if not isinstance(error_code, ErrorCodes):
            msg = 'Error code passed in the error_code param must be of type {0}'
            raise Error(ErrorCodes.ERR_INCORRECT_ERRCODE, msg, args=[ErrorCodes.__class__.__name__])
        # Storing the error code on the exception object
        self.error_code = error_code
        # storing the traceback which provides useful information about where the exception occurred
        self.traceback = traceback.format_exc()
        self.stack_trace = stack_trace if stack_trace is not None else ""
        # Prefixing the error code to the exception message
        try:
            msg = f"{str(message)} \n error code : {str(self.error_code)} \n " \
                  f"trace back: {str(self.traceback)} \n" \
                  f" stack trace: {str(self.stack_trace)} \n" \
                  f" {str(args)} \n str{str(kwargs)}"
        except (IndexError, KeyError):
            msg = f"{error_code.name},  {message}"
        super().__init__(msg)


@unique
class ErrorCodes(Enum):
    """Error codes for all module exceptions
    """

    ER_DEFAULT = auto()
    """  error code passed is not specified in enum ErrorCodes """
    ERR_INCORRECT_ERRCODE = auto()
    """ happens if a raster or a vector is not
     geo referenced """
    ERR_COORDINATE_REFERENCE_SYSTEM = auto()
    """ happens if a raster or a vector has a driver incompatibility with Odeon"""
    ERR_DRIVER_COMPATIBILITY = auto()
    """ happens if we ask or try to access to a non existent band of a raster with Odeon"""
    ERR_RASTER_BAND_NOT_EXIST = auto()
    """ happens if a file (in a json configuration CLI mostly) doesn't exist """
    ERR_FILE_NOT_EXIST = auto()
    """ happens if a dir (in a json configuration CLI mostly) doesn't exist """
    ERR_DIR_NOT_EXIST = auto()
    """ happens when the opening of a file raises an IO error """
    ERR_IO = auto()
    """ happens when a json schema validation raises an error"""
    ERR_JSON_SCHEMA_ERROR = auto()
    """ happens when something goes wrong in generation """
    ERR_GENERATION_ERROR = auto()
    """ happens when something goes wrong in sampling """
    ERR_SAMPLING_ERROR = auto()
    """ happens when something goes wrong in main configuration """
    ERR_MAIN_CONF_ERROR = auto()
    """ happens when a field is not found in any type of key value pair object """
    ERR_FIELD_NOT_FOUND = auto()
    """ happens when a critical test of interection returns false"""
    ERR_INTERSECTION_ERROR = auto()
    """ happens when an iterable object must be not empty"""
    ERR_EMPTY_ITERABLE_OBJECT = auto()
    """happens when an index is out of the bound of an object"""
    ERR_OUT_OF_BOUND = auto()
    """ happens when a path of datset is not valid"""
    INVALID_DATASET_PATH = auto()
    """ happens when something went wrong during the detection """
    ERR_DETECTION_ERROR = auto()
    """ happens when something went wrong during training """
    ERR_TRAINING_ERROR = auto()
    """ happens when we try to build a pytorch model"""
    ERR_MODEL_ERROR = auto()


    def __str__(self):
        """

        Returns
        -------
        str, str
         name of enum member, value of enum member
        """
        return f"name of error: {self.name}, code value of error: {self.value}"


def create_folder(path):
    """create folder with the whole hierarchy if required
    Parameters
    ----------
     path: complete path of the folder

    Returns
    -------
     None
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def files_exist(list_of_file):
    """

    Parameters
    ----------
    list_of_file : list

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_FILE_NOT_EXIST

    """

    for element in list_of_file:
        if isinstance(element, str):
            if os.path.isfile(element) is not True:
                raise Error(ErrorCodes.ERR_FILE_NOT_EXIST,
                                 f"the file {element} doesn't exists")
        else:
            for sub_element in element:
                if os.path.isfile(sub_element) is not True:
                    raise Error(ErrorCodes.ERR_FILE_NOT_EXIST,
                                     f"the file {sub_element} doesn't exists")


def dirs_exist(list_of_dir):
    """

    Parameters
    ----------
    list_of_dir : list

    Returns
    -------

    Raises
    -------
    odeon.commons.exception.OdeonError
     error code ERR_DIR_NOT_EXIST

    """

    for dir_name in list_of_dir:
        if os.path.isdir(dir_name) is not True:
            raise Error(ErrorCodes.ERR_DIR_NOT_EXIST,
                             f"the dir {dir_name} doesn't exists")
