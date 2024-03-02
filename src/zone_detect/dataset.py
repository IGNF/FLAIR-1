import numpy as np
import torch 
import rasterio.windows
import logging 

from torch.utils.data import Dataset
from skimage.util import img_as_float
from rasterio.enums import Resampling

LOGGER = logging.getLogger(__name__)


def convert(img, img_type):
    if img_type == "class_prob":
        if img.max() > 1:
            info = np.iinfo(img.dtype)  # get input datatype information
            img = img.astype(np.float32) / info.max  # normalize [0,1]
        img = np.iinfo(np.uint8).max * img  # scale by 255
        return img.astype(np.uint8)
    elif img_type == "argmax":
        img = np.argmax(img, axis=0)
        return np.expand_dims(img.astype(np.uint8), axis=0)
    else:
        LOGGER.warning("The output type has not been interpreted.")
        return img 


class Sliced_Dataset(Dataset):
    
    def __init__(self, 
                 dataframe,
                 img_path,
                 resolution,
                 bands,
                 patch_detection_size,
                 norma_dict
                ):
        
        
        self.dataframe = dataframe
        self.img_path = img_path
        self.resolution = resolution
        self.bands = bands
        self.height, self.width = patch_detection_size, patch_detection_size
        self.norma_dict = norma_dict[0]
        
        self.big_image = rasterio.open(self.img_path)   
        
    def __len__(self):
        return len(self.dataframe)  


    def close_raster(self):
        if self.big_image and not self.big_image.closed:
            self.big_image.close()  
            
            
    def normalization(self, in_img: np.ndarray, norm_type: str, means: list, stds: list):
        if norm_type == 'custom':
            if len(means) != len(stds):
                print("If custom, provided normalization means and stds should be of the same length. Going with scaling.")
                in_img = img_as_float(in_img)
            else:
                in_img = in_img.astype(np.float64)
                for i in range(in_img.shape[0]):
                    in_img[i] -= means[i]
                    in_img[i] /= stds[i]
        elif norm_type == 'scaling':
            in_img = img_as_float(in_img)
        else:
            print("Normalization argument should be 'scaling' or 'custom'. Going with scaling.")
            in_img = img_as_float(in_img)

        return in_img        


    def __getitem__(self, index):
        try: 
            bounds = self.dataframe.at[index, 'geometry'].bounds
            src = self.big_image

            window = rasterio.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.meta["transform"])
            patch_img = src.read(indexes=self.bands, window=window, out_shape=(len(self.bands), self.height, self.width),
                           resampling=Resampling.bilinear, boundless=True,
                         )

            patch_img = self.normalization(patch_img, self.norma_dict['norm_type'], self.norma_dict['norm_means'], self.norma_dict['norm_stds'])

            return {
                "image": torch.as_tensor(patch_img, dtype=torch.float),
                "index": torch.from_numpy(np.asarray([index])).int()
            }            
                     
        except rasterio._err.CPLE_BaseError as error:
            LOGGER.warning(f"CPLE error {error}")    