import torch
from torchmetrics import MeanMetric, JaccardIndex
import pytorch_lightning as pl


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criterion,
        optimizer,
        use_metadata=False,
        scheduler=None,
    ):

        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_metadata=use_metadata


    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None

            self.train_metrics = JaccardIndex(
                    task='multiclass',
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.val_metrics = JaccardIndex(
                    task='multiclass',
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = JaccardIndex(
                    task='multiclass',
                    num_classes=self.num_classes,
                    absent_score=1.0,
                    reduction='elementwise_mean')
            self.val_loss = MeanMetric()

    def forward(self, input_im, input_met):
        logits = self.model(input_im, input_met)
        return logits

    def step(self, batch):
        if self.use_metadata == True:
            images, metadata, targets = batch["img"], batch["mtd"], batch["msk"]
        else:
            images, metadata, targets = batch["img"], '', batch["msk"]            
        logits = self.forward(images, metadata)
        loss = self.criterion(logits, targets)

        with torch.no_grad():
            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)
            targets = torch.argmax(targets, dim=1)
            preds = preds.flatten(start_dim=1)  # Change shapes and cast target to integer for metrics computation
            targets = targets.flatten(start_dim=1).type(torch.int32)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_step_end(self, step_output):
        loss, preds, targets = (
            step_output["loss"].mean(),
            step_output["preds"],
            step_output["targets"]
        )
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        return loss

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log(
                "train_loss",
                self.train_epoch_loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                rank_zero_only=True
                )
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step_end(self, step_output):
        loss, preds, targets = (
            step_output["loss"].mean(),
            step_output["preds"],
            step_output["targets"]
        )
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.log(
            "val_miou",
            self.val_epoch_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True)
        self.val_loss.reset()
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.use_metadata == True:
            logits = self.forward(batch["img"], batch["mtd"])
        else:
            logits = self.forward(batch["img"], '')
        proba = torch.softmax(logits, dim=1)
        batch["preds"] =  torch.argmax(proba, dim=1)
        return batch

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
