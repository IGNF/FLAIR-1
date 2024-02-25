import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class metadata_mlp(nn.Module):
    """ 
    Light MLP to encode metadata
    
    """
    def __init__(self):
        super(metadata_mlp, self).__init__()

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


class smp_unet_mtd(nn.Module):
    """ 
    Pytorch segmentation U-Net with ResNet34 (default) 
    with added metadata information at encoder output
    
    """    
    def __init__(self,
                 architecture : str,
                 encoder : str ,
                 n_channels : int, 
                 n_classes : int,
                 use_metadata : bool = False
                 ):
        
        super(smp_unet_mtd, self).__init__()
        
        self.seg_model = smp.create_model(arch=architecture, 
                                          encoder_name=encoder, 
                                          classes=n_classes, 
                                          in_channels=n_channels,
         )
        
        self.use_metadata = use_metadata

        if use_metadata == True:
            self.enc = metadata_mlp()
            
    
    def forward(self, 
                x, 
                met,
                ):
        
        if self.use_metadata == True:
            feats = self.seg_model.encoder(x)
            x_enc = self.enc(met)
            x_enc = x_enc.unsqueeze(1).unsqueeze(-1).repeat(1,512,1,16)              
            feats[-1] = torch.add(feats[-1], x_enc)     
            output = self.seg_model.decoder(*feats)
            output = self.seg_model.segmentation_head(output)
        else:
            output = self.seg_model(x)

        return output