import torch
from torch import nn
#from torchinfo import summary 
#import sample.py
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 

    def __init__(self,in_channels: int=3,patch_size:int=16,embedding_dim:int=768):

        super().__init__()

        self.patch_size = patch_size

        #layer to convert image to patches
        self.patcher= nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        #layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)
     
    

    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on t


#rand_image_tensor = torch.randn(1, 3, 224, 224) # (batch_size, color_channels, height, width)


transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                       nhead=12,
                                                       dim_feedforward=3072,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)
#print(transformer_encoder_layer)


transformer_encoder = nn.TransformerEncoder(
    encoder_layer=transformer_encoder_layer,
    num_layers=12)


class ViT(nn.Module):
    def __init__(self,config):
            super().__init__()  
            img_size=config.img_size
            res_img_size = config.resize
            num_channels= config.num_channels
            patch_size= config.patch_size
            embedding_dim= config.embedding_dim
            dropout= config.dropout
            mlp_size= config.mlp_size
            num_transformer_layers= config.num_transformer_layers
            num_heads= config.num_heads
            num_classes= config.num_classes

            assert res_img_size % patch_size == 0, "Image size must be divisble by patch size." 

            #patch embedding
            self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

            #class token
            self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

            #positional embedding
            num_patches = (res_img_size * res_img_size) // patch_size**2  # N = HW/P^2
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

            #dropout
            self.embedding_dropout = nn.Dropout(p=dropout)

            #Trnasfomer encoder layers
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu",
                                                                                              batch_first=True,
                                                                                              norm_first=True), # Create a single Transformer Encoder Layer
                                                     num_layers=num_transformer_layers) # Stack it N times
            
 
            #MLP head
            self.mlp_head = nn.Sequential(
                                nn.LayerNorm(normalized_shape=embedding_dim),
                                nn.Linear(in_features=embedding_dim,
                                out_features=num_classes)
                            )


    def forward(self, x):
        
        batch_Size = x.shape[0]

        

        x = self.patch_embedding(x)
        class_token = self.class_token.expand(batch_Size,-1,-1)
        x =  torch.cat((class_token,x), dim=1)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x= self.transformer_encoder(x)
        x= self.mlp_head(x[:,0])

        return x



          