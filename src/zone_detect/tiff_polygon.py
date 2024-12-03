import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import MultiPolygon, Polygon
import numpy as np


def simplify_polygon(shapely_polygon):
    """ Return the polygon without hole. 
        If a Multipolygon, return the convex hull of the geometry.
    
    Args:
        shapely_polygon: a shapely Polygon or MultiPolygon

    Return:
        a MultiPolygon

    """
    if isinstance(shapely_polygon, MultiPolygon):
        # Get the smallest enclosing polygon (convex hull)
        # it's not perfect but still better than the rectangle
        shapely_polygon = shapely_polygon.convex_hull

    elif isinstance(shapely_polygon, Polygon):
        # we remove the holes inside the polygon
        shapely_polygon = Polygon(shapely_polygon.exterior.coords)

    else:
        raise ValueError("shapely_polygon must be a Polygon or a MultiPolygon. You Provided: {}".format(str(type(shapely_polygon))))

    return shapely_polygon



def retrieve_boundary_polygon_from_tif(img_path, simplify = True):

    # Open the tiff
    with rasterio.open(img_path) as src:

        # Use the lowest-resolution overview (for faster processing)
        if src.overviews(1):  # Check if overviews are available
            
            # Get the lowest-resolution overview (last in the list)
            overviews = src.overviews(1)
            overview_level = overviews[-1]
            print(f"I have the {overviews} overviews")

            # Read the downsampled data using the overview
            data = src.read(
                1,
                out_shape=(
                    1,
                    int(src.height / overview_level),
                    int(src.width / overview_level),
                ),
                resampling=Resampling.nearest,
            )

            # Update the transform for the overview resolution
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[1]), (src.height / data.shape[0])
            )

        else:
            print("no overview")
            # If no overviews, fallback to reading the full resolution (slower)
            data = src.read(1)
            transform = src.transform

        # Generate a binary mask for valid data (e.g., exclude zeros)
        valid_data_mask = np.where(data != 0, 1, 0).astype(np.uint8)

        # Extract shapes from the valid data mask
        shapes_gen = shapes(valid_data_mask, transform=transform)

        # Collect polygons where the mask is valid
        polygons = [shape(geom) for geom, value in shapes_gen if value == 1]

        # Merge all polygons into a single shape
        contour = unary_union(polygons)

        if simplify:
            contour = simplify_polygon(contour)

    return contour