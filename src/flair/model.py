import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoConfig


class FLAIR_ModelFactory(nn.Module):
    """
    A factory class for creating models based on the provided configuration.

    This class supports models from both SegmentationModelsPytorch and HuggingFace. 
    It also supports the use of metadata in the SegmentationModelsPytorch models.

    Attributes:
        model_provider (str): The provider of the model ('SegmentationModelsPytorch' or 'HuggingFace').
        use_metadata (bool): Whether to use metadata in the model.
        seg_model (nn.Module): The segmentation model.
        enc (MetadataMLP, optional): The metadata MLP encoder. Only used if `use_metadata` is True and `model_provider` is 'SegmentationModelsPytorch'.
    """
    def __init__(self,
                 config,
                 ):
        
        super(FLAIR_ModelFactory, self).__init__()

        self.model_provider = config['model_framework']['model_provider']
        self.use_metadata = config['use_metadata']
        
        n_channels = int(len(config['channels']))
        n_classes = int(len(config["classes"]))

        if self.use_metadata and model_provider == 'SegmentationModelsPytorch':
            self.enc = MetadataMLP()
            
        if self.model_provider == 'SegmentationModelsPytorch':
            encoder, architecture = config['model_framework']['SegmentationModelsPytorch']['encoder_decoder'].split('_')
            self.seg_model = smp.create_model(arch=architecture, 
                                              encoder_name=encoder, 
                                              classes=n_classes, 
                                              in_channels=n_channels,
             )  
            
        elif self.model_provider == 'HuggingFace':
            cfg_model = AutoConfig.from_pretrained(config['model_framework']['HuggingFace']['org_model'], 
                                                     num_labels=n_classes,
            )
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(config['model_framework']['HuggingFace']['org_model'], 
                                                                          config=cfg_model, 
                                                                          ignore_mismatched_sizes=True,
            )            

    def forward(self, x, met=None):

        if self.model_provider == 'SegmentationModelsPytorch':
        
            if self.use_metadata == True and self.model_provider == 'SegmentationModelsPytorch':
                feats = self.seg_model.encoder(x)
                x_enc = self.enc(met)
                x_enc = x_enc.unsqueeze(1).unsqueeze(-1).repeat(1,512,1,16)              
                feats[-1] = torch.add(feats[-1], x_enc)     
                output = self.seg_model.decoder(*feats)
                output = self.seg_model.segmentation_head(output)
            else:
                output = self.seg_model(x)
                
        elif self.model_provider == 'HuggingFace':
            output = self.seg_model(x) 
            output = output.logits

        return output



class MetadataMLP(nn.Module):
    """ 
    Light MLP to encode metadata
    
    """
    def __init__(self):
        super(MetadataMLP, self).__init__()

        self.enc_mlp = nn.Sequential(
            nn.Linear(45, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.4),
            nn.ReLU()        
        )            
        
    def forward(self, x):
        x = self.enc_mlp(x)
        return x    


