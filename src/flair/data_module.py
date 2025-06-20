from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.flair.data_loader import fit_dataset, predict_dataset

class flair_datamodule(LightningDataModule):

    def __init__(
        self,
        dict_train : dict = None,
        dict_val : dict = None,
        dict_test : dict = None,
        num_workers : int = 1,
        batch_size : int = 2,
        drop_last : bool = True,
        num_classes : int = 13,
        start_one : bool = True, 
        channels : list = [1,2,3,4,5],
        use_metadata: bool = True,
        use_augmentations : bool = True,
        norm_type : str = 'scaling',
        means : list = [],
        stds : list = []
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.start_one = start_one
        self.channels = channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations
        self.norm_type = norm_type
        self.means = means
        self.stds = stds

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = fit_dataset(
                dict_files=self.dict_train,
                channels=self.channels,
                num_classes=self.num_classes,
                start_one = self.start_one,
                use_metadata=self.use_metadata,
                use_augmentations=self.use_augmentations,
                norm_type= self.norm_type,
                means=self.means,
                stds=self.stds
            )

            self.val_dataset = fit_dataset(
                dict_files=self.dict_val,
                channels=self.channels,
                num_classes=self.num_classes,
                start_one = self.start_one,
                use_metadata=self.use_metadata,
                norm_type= self.norm_type,
                means=self.means,
                stds=self.stds
            )

        elif stage == "predict":
            self.pred_dataset = predict_dataset(
                dict_files=self.dict_test,
                channels=self.channels,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                norm_type= self.norm_type,
                means=self.means,
                stds=self.stds
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
            drop_last=False,
        )
