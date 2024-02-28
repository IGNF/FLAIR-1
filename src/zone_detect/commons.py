import os
from typing import TypeAlias, Literal, List, Dict, Tuple, Any

GEO_FLOAT_TUPLE: TypeAlias = Tuple[float, float]
GEO_INT_TUPLE: TypeAlias = Tuple[int, int]
OUTPUT_TYPE: TypeAlias = Literal['uint8', 'float32', 'bit', 'argmax']
PARAMS: TypeAlias = Dict[str, Any]

def files_exist(list_of_file):
    """
    Parameters
    ----------
    list_of_file : list
    Returns
    -------
    Raises
    -------
    Eception
    """

    for element in list_of_file:
        if isinstance(element, str):
            if not os.path.isfile(element):
                raise Exception("The file does not exist: " + element)
        else:
            for sub_element in element:
                if os.path.isfile(sub_element) is not True:
                    raise Exception("The file does not exist: " + element)


def dirs_exist(list_of_dir):
    """
    Parameters
    ----------
    list_of_dir : list
    Returns
    -------
    Raises
    -------
    Exception
    """

    for dir_name in list_of_dir:
        if os.path.isdir(dir_name) is not True:
            raise Exception(f"the dir {dir_name} doesn't exists")
