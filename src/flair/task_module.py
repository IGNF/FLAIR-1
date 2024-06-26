import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.aggregation import MeanMetric
import pytorch_lightning as pl



class segmentation_task_training(pl.LightningModule):
    def __init__(
        self,
        model,
        class_infos : dict,
        criterion = None,
        optimizer = None,
        use_metadata : bool = False,
        scheduler : bool = None,
    ):

        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_metadata=use_metadata

        self.num_classes = len(class_infos)
        self.class_names = [class_infos[i][1] for i in class_infos]
        self.class_weigths = [class_infos[i][0] for i in class_infos]

        
    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None

            self.train_metrics = MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average='weighted'
            )
            self.val_metrics = MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average='weighted'
            )

            self.val_iou = MulticlassJaccardIndex(
                    num_classes=self.num_classes, 
                    average=None
            )

            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    average='weighted'
            )
            self.val_loss = MeanMetric()

    def forward(self, input_im, input_met):
        logits = self.model(input_im, input_met)
        return logits

    def step(self, batch):
        if self.use_metadata:
            images, metadata, targets = batch["img"], batch["mtd"], batch["msk"]
        else:
            images, metadata, targets = batch["img"], '', batch["msk"]            
        logits = self.forward(images, metadata)
        targets = torch.argmax(targets, dim=1)
        loss = self.criterion(logits, targets)

        with torch.no_grad():
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            #targets = torch.argmax(targets, dim=1)
            preds = preds.flatten(start_dim=1)  # Change shapes and cast target to integer for metrics computation
            targets = targets.flatten(start_dim=1).type(torch.int32)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss

    def on_train_epoch_end(self):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log(
                "train_loss",
                self.train_epoch_loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                rank_zero_only=True,
                sync_dist=True,
                )
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        self.val_iou(preds=preds, target=targets)
        return loss

    def on_validation_epoch_end(self):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        iou_per_class = self.val_iou.compute()

        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        self.log(
            "val_miou",
            self.val_epoch_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        # Log IoU for each class
        for class_name, class_weight, iou in zip(self.class_names, self.class_weigths, iou_per_class):
            if class_weight == 0 :
                pass
            else:
                self.log(f"val_iou_{class_name}", 
                         iou.item(), 
                         on_step=False, 
                         on_epoch=True, 
                         prog_bar=False, 
                         logger=True,
                        rank_zero_only=True,
                        sync_dist=True,
                )

        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_iou.reset()

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
                "name": "Scheduler"
            }
            config = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config
        else: return self.optimizer 





class segmentation_task_predict(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes : int,
        use_metadata : bool = False,
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.use_metadata=use_metadata

    def forward(self, input_im, input_met):
        logits = self.model(input_im, input_met)
        return logits

    def step(self, batch):
        if self.use_metadata == True:
            images, metadata, targets = batch["img"], batch["mtd"], batch["msk"]
        else:
            images, metadata, targets = batch["img"], '', batch["msk"]            
        logits = self.forward(images, metadata)

        with torch.no_grad():
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            targets = torch.argmax(targets, dim=1)
            preds = preds.flatten(start_dim=1)  # Change shapes and cast target to integer for metrics computation
            targets = targets.flatten(start_dim=1).type(torch.int32)
        return preds, targets

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.use_metadata == True:
            logits = self.forward(batch["img"], batch["mtd"])
        else:
            logits = self.forward(batch["img"], '')
        proba = torch.softmax(logits, dim=1)
        batch["preds"] =  torch.argmax(proba, dim=1)
        return batch
     

