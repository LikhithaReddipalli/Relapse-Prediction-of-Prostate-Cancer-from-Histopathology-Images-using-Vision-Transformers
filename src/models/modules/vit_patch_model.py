from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
from torchvision.models import vit_b_16 , vit_l_16
import torchvision
import torch
import torch.nn as nn
import torch.fx
import configs.config as config
from torchvision.models.feature_extraction import create_feature_extractor
from collections import OrderedDict


class CustomModelViT(nn.Module):
    def __init__(self,config):
        super(CustomModelViT, self).__init__()


        # Load the pretrained ViT model only once
        pretrained_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(config.device)

        #pretrained_model = vit_b_16().to(config.device)
        pretrained_model.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)
        
        
        #pretrained_model.load_state_dict(torch.load(config.checkpoint))

        # Replace the classification head
        #pretrained_model.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)

        


        self.feature_extractor = create_feature_extractor(pretrained_model, return_nodes=['encoder.ln'])#

        self.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)

        for parameter in self.heads.parameters():
            parameter.requires_grad = True

        
    def forward(self, x):

        output1 = self.feature_extractor(x)['encoder.ln']
        output1 = output1[:, 0, :] #.shape
        output1 = torch.reshape(output1,(-1,16,1,768))
        
        
        output1 = torch.mean(output1, dim=1, keepdim=True)


        #max_value, max_index = torch.max(output1, dim=1, keepdim = True) #uncomment this for max pooling
        #output1 = max_value
       
        output = self.heads(output1)

        return output
    




class CustomModelViT_Test(nn.Module):
    def __init__(self,config):
        super(CustomModelViT_Test, self).__init__()


        self.checkpoint = '/data/PANDA/code/survival_prediction/Experiments/2023_08_31_11_42_57/checkpoints_50.pth' 


        old_suffix = "pretrained_model." 
        #old_suffix1 = "feature_extractor."
        new_prefix = ""


        checkpoint = torch.load(self.checkpoint, map_location='cuda:1')  # Adjust the map_location as needed

        # Remove the "pretrained_model" suffix from keys and create a new state_dict
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace(old_suffix, new_prefix)
            new_state_dict[new_key] = value
        
        '''new_state_dict = {}
        prefix = 'pretrained_model.', 'feature_extractor.'
        for key, value in checkpoint.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]  # Remove the prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value'''

        # Load the modified state_dict into your model
        #self.load_state_dict(new_state_dict)

        self.pretrained_model = vit_b_16().to(config.device)
        self.pretrained_model.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)

        #self.pretrained_model.load_state_dict(torch.load(self.checkpoint))

        self.pretrained_model.load_state_dict(new_state_dict)   


        self.feature_extractor = create_feature_extractor(self.pretrained_model, return_nodes=['encoder.ln'])


        #self.pretrained_model.heads= nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)


        
    def forward(self, x):

        # Pass the input through the pretrained model

        #features = self.features(x)

        # Apply pooling
        #features_pooled2 = self.pooling_layer2(features_pooled1.transpose(1, 0).squeeze())
        #output = self.features(x)
        #output2 = self.features2(output)
        # Apply linear layer
        output1 = self.feature_extractor(x)['encoder.ln']
        output1 = output1[:, 0, :] #.shape
        output1 = torch.reshape(output1,(-1,16,1,768))
        
        
        #output1 = torch.mean(output1, dim=1, keepdim=True)


        max_value, max_index = torch.max(output1, dim=1, keepdim = True)
        output1 = max_value
       
        output = self.pretrained_model.heads(output1)

        #torch.mean(output1, dim=0, keepdim=True).shape
        #pooled_features = self.pooling(output1)

        # Flatten the pooled features for the classification head
        #pooled_features = torch.flatten(pooled_features, 1)

        #index = torch.argmax(output1[:, 0, :].mean(dim=1))
        #row = output1[:, 0, :][index]

        #1D convolution layers for input size 1, 16*768 -> 1, 1

        #max_value, max_index = torch.max(output1[:, 0, :], dim=0)
        #output_temp = output1[max_index]

        
        #output = self.linear_layer(torch.tensor(max_value.unsqueeze(0)))

        return output
    











