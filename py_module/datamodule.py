from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import Fit_Dataset, Predict_Dataset


class OCS_DataModule(LightningDataModule):

    def __init__(
        self,
        dict_train=None,
        dict_val=None,
        dict_test=None,
        num_workers=1,
        batch_size=2,
        drop_last=True,
        num_classes=13,
        num_channels=5,
        use_metadata=True,
        use_augmentations=True
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                use_augmentations=self.use_augmentations
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )