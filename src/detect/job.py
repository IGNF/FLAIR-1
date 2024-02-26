"module of Jobs classes, typically detection jobs"
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

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="detection_job.csv"):
        """Job class used for patch based detection
        It simply encapsulates a pandas.DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            a pandas dataframe with at least a job_done field
        path : str
            path to save the job
        recover : bool, optional
            if set to True it will see if a previous job has been
            saved and will start from its ending point, by default False
        file_name : str, optional
            file name of the saved job, by default "detection_job.csv"
        """

        self._df = df
        self._job_done = None
        self._path = path
        self._job_file = os.path.join(self._path, file_name)
        self._recover = recover

        if self._recover and os.path.isfile(self._job_file):

            self._df = self.read_file(self._job_file)

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
                 recover=False,
                 file_name="detection_job.shp"):
        """Job class used for zone based detection with output by dalle
        It simply encapsulates a geopandas.DataFrame

        Parameters
        ----------
        df : gpd.DataFrame
            a geopandas dataframe with at least a job_done field and a geometry
        path : str
            path to save the job
        recover : bool, optional
            if set to True it will see if a previous job has been
            saved and will start from its ending point, by default False
        file_name : str, optional
            file name of the saved job, by default "detection_job.csv"
        """
        super(ZoneDetectionJob, self).__init__(df, path, recover=recover, file_name=file_name)

    def __str__(self):
        return f" Zone Job Detection with dataframe {self._df}"

    @classmethod
    def read_file(cls, file_name):
        return gpd.read_file(file_name)

    def get_job_done(self):
        df_grouped = self._df.groupby(["output_id"]).agg({"job_done": ["sum", "count"]})
        LOGGER.debug(df_grouped)
        LOGGER.debug(df_grouped.index.values.tolist())
        LOGGER.debug(df_grouped.columns)
        job_done_id = df_grouped[df_grouped["job_done", "sum"] == df_grouped["job_done", "count"]].index.values.tolist()
        LOGGER.debug(job_done_id)
        LOGGER.debug(len(job_done_id))
        LOGGER.debug(self._df["output_id"].isin(job_done_id))
        return self._df[self._df["output_id"].isin(job_done_id)]

    def get_todo_list(self):
        if self._job_done is None:
            return self._df
        else:
            return self._df[~self._df["output_id"].isin(self._job_done["output_id"].values)]

    def save_job(self):
        out = self._df
        LOGGER.debug(f"length of job todo {len(out)}")
        if self._job_done is not None:
            out = pd.concat([out, self._job_done])
            LOGGER.debug(f"job done: {len(self._job_done)}")
        LOGGER.debug(f"lenght of total job {len(out)}")
        LOGGER.debug(out)
        LOGGER.debug(out.columns)
        LOGGER.debug(out.dtypes)
        LOGGER.debug(out["id"])
        out.to_file(self._job_file)

    def get_bounds_at(self, idx):
        LOGGER.debug(f"index {idx}")
        LOGGER.debug(f"indices:\n {self._df.index.values.tolist()}")
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
    def build_job(gdf, output_size, resolution, overlap=0, out_dalle_size=None):
        """Helper function to build a job detection by zone, aggregated or not by dalle.
        It will slice each entry rectangle polygon based on their bounds, with an overlapping factor.
        If out_dalle_size is set, it will first slice by dalle size, and then slice by patch size
        with an overlaping factor.

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
            the overlapping factor, by default 0
        out_dalle_size : Union[int, float], optional
            used to build job with aggregated patches by dalle, by default None

        Returns
        -------
        tuple[gdp.GeoDaTaFrame, gdp.GeoDaTaFrame]
            a geodataframe with the list of polygon patches generated by slicing.
            a geoDataframe with a list container polygon, the original polygons of entry
            or the generated dalle it output_dalle_size has been set.
        """

        output_size_u = [
            output_size * resolution[0],
            output_size * resolution[1]
        ]
        overlap_u = [
            overlap * resolution[0],
            overlap * resolution[1]
        ]
        step = [
            output_size_u[0] - (2 * overlap_u[0]),
            output_size_u[1] - (2 * overlap_u[1])
        ]
        tmp_list = []
        write_gdf = None

        for idx, df_row in gdf.iterrows():
            if out_dalle_size is not None:
                bounds = df_row["geometry"].bounds
                min_x, min_y = bounds[0], bounds[1]
                max_x, max_y = bounds[2], bounds[3]
                name = df_row["id"] if "id" in gdf.columns else idx

                for i in np.arange(min_x, max_x, out_dalle_size[0]):
                    for j in np.arange(min_y, max_y, out_dalle_size[1]):
                        "handling case where the extent is not a multiple of step"
                        if i + out_dalle_size[0] > max_x:
                            i = max_x - out_dalle_size[0]
                        if j + out_dalle_size[1] > max_y:
                            j = max_y - out_dalle_size[1]
                        left = i
                        right = i + out_dalle_size[0]
                        bottom = j
                        top = j + out_dalle_size[1]
                        col, row = int((j - min_y) // resolution[0]) + 1, int((i - min_x) // resolution[1]) + 1
                        row_d = {
                                    "id": f"{name}-{row}-{col}",
                                    "name": name,
                                    "job_done": False,
                                    "left": left,
                                    "bottom": bottom,
                                    "right": right,
                                    "top": top,
                                    "affine": "",
                                    "patch_count": 0,
                                    "nb_patch_done": 0,
                                    "geometry": create_box_from_bounds(i,  i + out_dalle_size[0], j, j + out_dalle_size[1])
                                }
                        tmp_list.append(row_d)
            else:
                bounds = df_row["geometry"].bounds
                left, bottom = bounds[0], bounds[1]
                right, top = bounds[2], bounds[3]
                name = df_row["id"] if "id" in gdf.columns else idx
                col, row = int(bottom // resolution[0]), int(left // resolution[0])
                row_d = {
                    "id": f"{name}-{row}-{col}",
                    "name": name,
                    "job_done": False,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "affine": "",
                    "patch_count": 0,
                    "nb_patch_done": 0,
                    "geometry": df_row.geometry
                }
                tmp_list.append(row_d)
        write_gdf = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")
        tmp_list = []
        for idx, df_row in write_gdf.iterrows():

            bounds = df_row["geometry"].bounds
            print(f"row bounds {bounds}")
            print(f"overlap_u {overlap_u}")
            LOGGER.debug(bounds)
            min_x, min_y = bounds[0], bounds[1]
            max_x, max_y = bounds[2], bounds[3]

            for i in np.arange(min_x - overlap_u[0], max_x + overlap_u[0], step[0]):
                for j in np.arange(min_y - overlap_u[1], max_y + overlap_u[1], step[1]):
                    "handling case where the extent is not a multiple of step"
                    if i + output_size_u[0] > max_x + overlap_u[0]:
                        i = max_x + overlap_u[0] - output_size_u[0]
                    if j + output_size_u[1] > max_y + overlap_u[1]:
                        j = max_y + overlap_u[1] - output_size_u[1]
                    left = i + overlap_u[0]
                    right = i + output_size_u[0] - overlap_u[0]
                    bottom = j + overlap_u[1]
                    top = j + output_size_u[1] - overlap_u[1]
                    col, row = int((j - min_y) // resolution[0]) + 1, int((i - min_x) // resolution[1]) + 1
                    row_d = {
                                "id": str(f"{idx + 1}-{row}-{col}"),
                                "output_id": df_row["id"],
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
                                "geometry": create_box_from_bounds(i, i + output_size_u[0], j, j + output_size_u[1])
                            }
                    tmp_list.append(row_d)
                    if out_dalle_size is not None:
                        write_gdf.at[idx, "patch_count"] += 1
        gdf_output = gpd.GeoDataFrame(tmp_list, crs=gdf.crs, geometry="geometry")
        return gdf_output, write_gdf


class ZoneDetectionJobNoDalle(PatchJobDetection):
    """Job class used for zone based detection with output by patch
        It simply encapsulates a geopandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame, path, recover=False, file_name="detection_job.shp"):
        """Job class used for zone based detection with output by patch
        It simply encapsulates a geopandas.DataFrame

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            a geopandas dataframe with at least a job_done field and a geometry
        path : str
            path to save the job
        recover : bool, optional
            if set to True it will see if a previous job has been
            saved and will start from its ending point, by default False
        file_name : str, optional
            file name of the saved job, by default "detection_job.csv"
        """
        super(ZoneDetectionJobNoDalle, self).__init__(df, path, recover, file_name=file_name)

    @classmethod
    def read_file(cls, file_name):

        return gpd.read_file(file_name)

    def get_bounds_at(self, idx):
        LOGGER.debug(f"index {idx}")
        LOGGER.debug(f"indices:\n {self._df.index.values.tolist()}")
        return self.get_cell_at(idx, "geometry").bounds

    def save_job(self):

        out = self._df

        if self._job_done is not None:

            out = pd.concat([out, self._job_done])

        LOGGER.debug(len(out))
        LOGGER.debug(out)
        LOGGER.debug(out.columns)
        LOGGER.debug(out.dtypes)
        LOGGER.debug(out["id"])
        out.to_file(self._job_file)