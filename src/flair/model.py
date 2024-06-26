import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SwinModel, SwinConfig,  UperNetForSemanticSegmentation


class FLAIR_ModelFactory(nn.Module):
    """ 
    Pytorch segmentation U-Net with ResNet34 (default) 
    with added metadata information at encoder output
    
    """    
    def __init__(self,
                 architecture: str,
                 encoder: str,
                 n_channels: int, 
                 n_classes: int,
                 use_metadata: bool = False
                 ):
        
        super(FLAIR_ModelFactory, self).__init__()
        
        # Initialize the SMP model without specifying the encoder
        #self.seg_model = smp.create_model(
        #    arch=architecture, 
        #    encoder_name=encoder,  # We'll manually set the encoder
        #    classes=n_classes, 
        #    in_channels=n_channels,
        #)
        
        self.use_metadata = use_metadata

        if use_metadata:
            self.enc = MetadataMLP()
            



        config = UperNetForSemanticSegmentation.config_class.from_pretrained('openmmlab/upernet-swin-small', num_labels=n_classes)
        self.model = UperNetForSemanticSegmentation.from_pretrained('openmmlab/upernet-swin-small', config=config, ignore_mismatched_sizes=True)

 
    def forward(self, x, met=None):
        # Extract features from the encoder
        #encoder_outputs = self.encoder(x)
        #last_hidden_state = encoder_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        #batch_size, seq_length, hidden_size = last_hidden_state.shape
        #spatial_dim = int(seq_length**0.5)
        
        # Reshape to match expected input shape for the decoder
        #feats = last_hidden_state.permute(0, 2, 1).reshape(batch_size, hidden_size, spatial_dim, spatial_dim)

        # Simulate multi-scale features by interpolating the last hidden state
        #features = [
        #    nn.functional.interpolate(feats, size=(spatial_dim, spatial_dim), mode='bilinear', align_corners=False),  # original resolution
        #    nn.functional.interpolate(feats, size=(spatial_dim*2, spatial_dim*2), mode='bilinear', align_corners=False),  # 1/2 resolution
        #    nn.functional.interpolate(feats, size=(spatial_dim*4, spatial_dim*4), mode='bilinear', align_corners=False),   # 1/4 resolution
        #    nn.functional.interpolate(feats, size=(spatial_dim*8, spatial_dim*8), mode='bilinear', align_corners=False),   # 1/8 resolution
        #    nn.functional.interpolate(feats, size=(spatial_dim*16, spatial_dim*16), mode='bilinear', align_corners=False)  # 1/16 resolution
        #]
        
        #print(f"x shape: {x.shape}")
        #print(f"last_hidden_state shape: {last_hidden_state.shape}")
        #print(f"feats shape: {feats.shape}")
        #for i, feat in enumerate(features):
        #    print(f"features[{i}] shape: {feat.shape}")
       # 
       # # Decode the features
       #output = self.decoder(*features)
       #output = self.segmentation_head(output)

        output = self.model(x) 
        logits = output.logits

        return logits



class FLAIR_ModelFactory2(nn.Module):
    """ 
    PyTorch segmentation U-Net with ResNet34 (default) 
    with added metadata information at encoder output
    """    
    def __init__(self,
                 architecture: str,
                 encoder: str,
                 n_channels: int, 
                 n_classes: int,
                 use_metadata: bool = False
                 ):
        
        super(FLAIR_ModelFactory2, self).__init__()
        
        self.use_metadata = use_metadata

        if use_metadata:
            self.enc = MetadataMLP()
        
        # Load the pre-trained model
        config = UperNetForSemanticSegmentation.config_class.from_pretrained('openmmlab/upernet-swin-small', num_labels=n_classes, in_channels=n_channels)
        self.model = UperNetForSemanticSegmentation.from_pretrained('openmmlab/upernet-swin-small', config=config, ignore_mismatched_sizes=True)
        
        # Adjust the patch embeddings projection layer to accept 4 input channels
        self.model.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(in_channels=n_channels, 
                                                                                out_channels=self.model.backbone.embeddings.patch_embeddings.projection.out_channels, 
                                                                                kernel_size=self.model.backbone.embeddings.patch_embeddings.projection.kernel_size, 
                                                                                stride=self.model.backbone.embeddings.patch_embeddings.projection.stride, 
                                                                                padding=self.model.backbone.embeddings.patch_embeddings.projection.padding, 
                                                                                bias=self.model.backbone.embeddings.patch_embeddings.projection.bias is not None)

        # Modify the decode head's classifier
        self.model.decode_head.classifier = nn.Conv2d(in_channels=self.model.decode_head.classifier.in_channels, out_channels=n_classes, kernel_size=self.model.decode_head.classifier.kernel_size)

        # Modify the auxiliary head's classifier
        self.model.auxiliary_head.classifier = nn.Conv2d(in_channels=self.model.auxiliary_head.classifier.in_channels, out_channels=n_classes, kernel_size=self.model.auxiliary_head.classifier.kernel_size)

    def forward(self, x, met=None):
        if self.use_metadata and met is not None:
            met_features = self.enc(met)
            # Combine met_features with the encoder output or any other suitable layer
            # This step depends on how you want to incorporate metadata into the model
            # Example: concatenate metadata features with the encoder output
            # encoder_output = self.model.backbone(x)
            # combined = torch.cat((encoder_output, met_features), dim=1)
            # logits = self.model.decode_head(combined)
            # Return the logits

        output = self.model(x) 
        logits = output.logits

        return logits






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


