import numpy as np
import logging
import rasterio
from skimage import img_as_float

import torch
from torch.utils.data import Dataset


def norm(in_img : np.ndarray, 
         norm_type : str = None, 
         means : list = [], 
         stds: list = [],
         ):
    
    in_img = in_img.astype(np.float64) 
    if norm_type not in ['scaling','custom','without']:
            logging.error("Normalization argument should be 'scaling', 'custom' or 'without'.")
            raise SystemExit()
    if norm_type: 
        if norm_type == 'custom':
            if len(means) != len(stds):
                logging.error("If custom, provided normalization means and stds should be of same lenght.")
                raise SystemExit()                
            else:
                for i in range(len(means)):
                    in_img[i] -= means[i]
                    in_img[i] /= stds[i]
        elif norm_type == 'scaling':
            in_img = img_as_float(in_img)
    return in_img



class fit_dataset(Dataset):

    def __init__(self,
                 dict_files : dict,
                 channels : list = [1,2,3,4,5],
                 num_classes : int = 13, 
                 use_metadata : bool = True,
                 use_augmentations : bool = None,
                 norm_type : str = 'scale',
                 means : list = [],
                 stds : list = []
                 ):

        self.list_imgs = np.array(dict_files["IMG"])
        self.list_msks = np.array(dict_files["MSK"])
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD"])
        self.use_augmentations = use_augmentations
        self.channels = channels
        self.num_classes = num_classes
        self.norm_type= norm_type
        self.means = means
        self.stds = stds


    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            #array[array > self.num_classes] = self.num_classes
            array = array-1
            array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array  

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        
        mask_file = self.list_msks[index]
        msk = self.read_msk(raster_file=mask_file)

        if self.use_augmentations is not None:
            sample = {"image" : img.swapaxes(0, 2).swapaxes(0, 1), "mask": msk.swapaxes(0, 2).swapaxes(0, 1)}
            transformed_sample = self.use_augmentations(**sample)
            img, msk = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(), transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()            
        
        img = norm(img, norm_type=self.norm_type, means=self.means, stds=self.stds)

        if self.use_metadata == True:
            mtd = self.list_metadata[index]
            return {"img": torch.as_tensor(img, dtype=torch.float), 
                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
                    "msk": torch.as_tensor(msk, dtype=torch.float)}
        else:
            return {"img": torch.as_tensor(img, dtype=torch.float), 
                    "msk": torch.as_tensor(msk, dtype=torch.float)}  




class predict_dataset(Dataset):

    def __init__(self,
                 dict_files : dict,
                 channels : list = [1,2,3,4,5],
                 num_classes : int = 13, 
                 use_metadata : bool = True,
                 norm_type : str = 'scaling',
                 means : list = [],
                 stds : list = []
                 ):
        
        self.list_imgs = np.array(dict_files["IMG"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD"])
        self.channels = channels
        self.norm_type= norm_type
        self.means = means
        self.stds = stds

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
        return array
        
    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        
        img = norm(img, norm_type=self.norm_type, means=self.means, stds=self.stds)

        if self.use_metadata == True:
            mtd = self.list_metadata[index]
            return {"img": torch.as_tensor(img, dtype=torch.float), 
                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
                    "id": image_file}
        else:
           
            return {"img": torch.as_tensor(img, dtype=torch.float),
                    "id": image_file}  