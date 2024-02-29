from pickletools import uint8
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio
from PIL import Image


class predictionwriter(BasePredictionWriter):

    def __init__(
        self,
        config,
        output_dir,
        write_interval,
    ):
        super().__init__(write_interval)
        self.config = config
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        
        if self.config['georeferencing_output']:      
            preds, filenames = prediction["preds"], prediction["id"]
            preds = preds.cpu().numpy().astype('uint8') # Pass prediction on CPU
            
            for prediction, filename in zip(preds, filenames):
                output_file = str(self.output_dir+'/'+filename.split('/')[-1].replace(filename.split('/')[-1], 'PRED_'+filename.split('/')[-1]))
                with rasterio.open(filename, 'r') as f:
                    meta = f.profile  # extract georeferencing information from input img
                    meta['count'] = 1
                    meta['compress'] = 'lzw'
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(prediction, 1)
        else:
            preds, filenames = prediction["preds"], prediction["id"]
            preds = preds.cpu().numpy().astype('uint8')  # Pass prediction on CPU

            for prediction, filename in zip(preds, filenames):
                output_file = str(self.output_dir+'/'+filename.split('/')[-1].replace(filename.split('/')[-1], 'PRED_'+filename.split('/')[-1]))
                Image.fromarray(prediction).save(output_file,  compression='tiff_lzw') 


    def on_predict_batch_end(
            self, 
            trainer, 
            pl_module, 
            outputs, 
            batch, 
            batch_idx, 
            dataloader_idx = 0):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )