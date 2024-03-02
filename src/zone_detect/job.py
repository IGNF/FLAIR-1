import os
from logging import getLogger

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping

LOGGER = getLogger(__name__)


def create_polygon_from_bounds(x_min, x_max, y_min, y_max):
    return mapping(box(x_min, y_max, x_max, y_min))


def create_box_from_bounds(x_min, x_max, y_min, y_max):
    return box(x_min, y_max, x_max, y_min)


class BaseDetectionJob:
    pass


class PatchJobDetection(BaseDetectionJob):
    """
    Job class used for patch based detection
    It simply encapsulates a pandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame, path, file_name="detection_job.csv"):
        """Job class used for patch based detection
        It simply encapsulates a pandas.DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            a pandas dataframe with at least a job_done field
        path : str
            path to save the job
        file_name : str, optional
            file name of the saved job, by default "detection_job.csv"
        """

        self._df = df
        self._job_done = None
        self._path = path
        self._job_file = os.path.join(self._path, file_name)
        self.keep_only_todo_list()

    def __len__(self):
        return len(self._df)

    def __str__(self):
        return f" PatchJobDetection with dataframe {self._df}"

    @classmethod
    def read_file(cls, file_name):
        return pd.read_csv(file_name)

    def get_row_at(self, index):
        return self._df.iloc[index]

    def get_cell_at(self, index, column):
        return self._df.at[index, column]

    def set_cell_at(self, index, column, value):
        self._df.at[index, column] = value

    def get_job_done(self):
        return self._df[self._df["job_done"] == 1]

    def get_todo_list(self):
        return self._df[self._df["job_done"] == 0]

    def keep_only_todo_list(self):
        self._job_done = self.get_job_done()
        self._df = self.get_todo_list()
        self._df.reset_index(drop=True)
        self._job_done.reset_index(drop=True)

    def save_job(self):
        out = self._df
        if self._job_done is not None:
            out = pd.concat([out, self._job_done])
        out.to_csv(os.path.join(self._path, self._job_file))


class ZoneDetectionJob(PatchJobDetection):
    """Job class used for zone based detection with output by dalle
        It simply encapsulates a geopandas.DataFrame
    """
    def __init__(self,
                 df: pd.DataFrame,
                 path,
                 file_name="detection_job.shp"):
        
        """Job class used for zone based detection with output by dalle
        It simply encapsulates a geopandas.DataFrame

        Parameters
        ----------
        df : gpd.DataFrame
            a geopandas dataframe with at least a job_done field and a geometry
        path : str
            path to save the job
        file_name : str, optional
            file name of the saved job, by default "detection_job.csv"
        """
        super(ZoneDetectionJob, self).__init__(df, path, file_name=file_name)

    def __str__(self):
        return f" Zone Job Detection with dataframe {self._df}"

    @classmethod
    def read_file(cls, file_name):
        return gpd.read_file(file_name)

    def get_job_done(self):
        df_grouped = self._df.groupby(["output_id"]).agg({"job_done": ["sum", "count"]})
        job_done_id = df_grouped[df_grouped[("job_done", "sum")] == df_grouped[("job_done", "count")]].index.values.tolist()
        LOGGER.debug(f"Job done IDs: {job_done_id}, Count: {len(job_done_id)}")
        LOGGER.debug(f"Output IDs in job done: {self._df['output_id'].isin(job_done_id)}")
        return self._df[self._df["output_id"].isin(job_done_id)]

    def get_todo_list(self):
        if self._job_done is None:
            return self._df
        else:
            return self._df[~self._df["output_id"].isin(self._job_done["output_id"].values)]

    def save_job(self):
        out = self._df
        if self._job_done is not None:
            out = pd.concat([out, self._job_done])
            LOGGER.debug(f"Job done: {len(self._job_done)}")
        LOGGER.debug(f"Length of total job: {len(out)}\n{out}\nColumns: {out.columns}\nData types: {out.dtypes}\nIDs: {out['id']}")
        out.to_file(self._job_file, driver="GPKG")

    def get_bounds_at(self, idx):
        LOGGER.debug(f"Index {idx}\nIndices:\n{self._df.index.values.tolist()}")
        return self.get_cell_at(idx, "geometry").bounds

    def job_finished_for_output_id(self, output_id):
        dalle_df = self._df[self._df["output_id"] == output_id]
        if len(dalle_df[dalle_df["job_done"] == 1]) == len(dalle_df):
            return True
        else:
            return False

    def mark_dalle_job_as_done(self, output_id):
        LOGGER.debug(f"dalle {output_id} done")
        self._df.loc[self._df["output_id"] == output_id, "dalle_done"] = 1
        self._job_done = pd.concat([self._job_done, self._df[self._df["output_id"].isin([output_id])]])
        self._df = self._df[~self._df["output_id"].isin([output_id])]


    @staticmethod
    def build_job(gdf, patch_size, resolution, margin=0):
        """Helper function to build a job detection.
        It will slice the rectangle polygon based on bounds, with an overlapping factor.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            a geodataframe with a list of polygon
        output_size : int
            the size of window slice
        resolution: obj:`list` of :obj: `float`
            resolution, in the crs unit of the GeoPandasDataFrame.
            Used to slice polygon at specific resolution.
        overlap : int, optional
            the overlapping size in pixels

        Returns
        -------
        tuple[gdp.GeoDaTaFrame, gdp.GeoDaTaFrame]
            a geodataframe with the list of polygon patches generated by slicing.
        """

        geo_output_size = [
            patch_size * resolution[0],
            patch_size * resolution[1]
        ]
        geo_margin = [
            margin * resolution[0],
            margin * resolution[1]
        ]
        geo_step = [
            geo_output_size[0] - (2 * geo_margin[0]),
            geo_output_size[1] - (2 * geo_margin[1])
        ]

        tmp_list = []
        for idx, df_row in gdf.iterrows():
            bounds = df_row["geometry"].bounds
            LOGGER.debug(f'Bounds of polygon : {bounds}')
            min_x, min_y = bounds[0], bounds[1]
            max_x, max_y = bounds[2], bounds[3]

            for x_coord in np.arange(min_x - geo_margin[0], max_x + geo_margin[0], geo_step[0]):
                for y_coord in np.arange(min_y - geo_margin[1], max_y + geo_margin[1], geo_step[1]):

                    "handling case where the extent is not a multiple of geo_step"
                    if x_coord + geo_output_size[0] > max_x + geo_margin[0]:
                        x_coord = max_x + geo_margin[0] - geo_output_size[0]
                    if y_coord + geo_output_size[1] > max_y + geo_margin[1]:
                        y_coord = max_y + geo_margin[1] - geo_output_size[1]

                    left = x_coord + geo_margin[0]
                    right = x_coord + geo_output_size[0] - geo_margin[0]
                    bottom = y_coord + geo_margin[1]
                    top = y_coord + geo_output_size[1] - geo_margin[1]
                    col, row = int((y_coord - min_y) // resolution[0]) + 1, int((x_coord - min_x) // resolution[1]) + 1
                    row_d = {
                                "id": str(f"{idx + 1}-{row}-{col}"),
                                "output_id": df_row['out_name'],
                                "dalle_done": 0,
                                "job_done": 0,
                                "left": left,
                                "bottom": bottom,
                                "right": right,
                                "top": top,
                                "left_o": df_row["left"],
                                "bottom_o": df_row["bottom"],
                                "right_o": df_row["right"],
                                "top_o": df_row["top"],
                                "geometry": create_box_from_bounds(x_coord, x_coord + geo_output_size[0], y_coord, y_coord + geo_output_size[1])
                            }
                    tmp_list.append(row_d)

        gdf_output = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")

        return gdf_output
        
