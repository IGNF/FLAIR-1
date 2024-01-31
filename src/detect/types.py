from typing import TypeAlias, Literal, List, Dict, Tuple, Any

GEO_FLOAT_TUPLE: TypeAlias = Tuple[float, float]
GEO_INT_TUPLE: TypeAlias = Tuple[int, int]
OUTPUT_TYPE: TypeAlias = Literal['uint8', 'float32', 'bit', 'argmax']
PARAMS: TypeAlias = Dict[str, Any]
