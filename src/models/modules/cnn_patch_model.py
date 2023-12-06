from torchvision.models import efficientnet_b0 , EfficientNet_B0_Weights
import torch
import torch.nn as nn
import configs.config as config



# Custom model of CNN combined with multiple instance learning for relapse prediction  
class CustomModel(nn.Module):
    def __init__(self,config):
        super(CustomModel, self).__init__()

        #pretrained_model = efficientnet_b0(weights= EfficientNet_B0_Weights.DEFAULT)#.to(config.device)

        pretrained_model = efficientnet_b0() #.to(config.device)

        pretrained_model.classifier = nn.Sequential(
                                                nn.Dropout(p=0.2, inplace=True),
                                                nn.Linear(in_features=1280, out_features= config.num_classes, bias=True)
                                                )

        pretrained_model.load_state_dict(torch.load(config.checkpoint, map_location='cuda:1')) #.to(config.device)

        for parameter in pretrained_model.classifier.parameters():
            parameter.requires_grad = True

        self.features = nn.Sequential(*list(pretrained_model.features))      
        
        # Define the pooling layer
        self.pooling_layer1 = nn.AdaptiveAvgPool2d(1)#.to(config.device)
        self.pooling_layer2 = nn.AdaptiveAvgPool1d(1)#.to(config.device)
        
        # Define the linear layer
        self.linear_layer = nn.Linear(1280,1)#.to(config.device)
        self.linear_layer.requires_grad = True
        
    def forward(self, x):

        # Pass the input through the pretrained model

        features = self.features(x)

        #classifier = self.classifier(features)
        features_pooled1 = self.pooling_layer1(features)
        features_pooled1 = torch.reshape(features_pooled1,(-1,16,1,1280))

        max_value, max_index = torch.max(features_pooled1, dim=1, keepdim = True)
        output1 = max_value

        #output1 = torch.mean(features_pooled1, dim=1, keepdim=True) # Uncomment this for mean pooling
       
          
        # Apply linear layer
        output = self.linear_layer(output1).unsqueeze(dim=2)
        
        return output
    

