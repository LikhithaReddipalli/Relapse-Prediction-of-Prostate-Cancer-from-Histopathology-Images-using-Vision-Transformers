import sys
sys.path.append('C:\\Users\\likhi\\Desktop\\Thesis\\pc\\pc_code\\survival_prediction')
from src.models.hier_pretraining_chen.vision_transformer import vit_base, vit_small
from src.models.hier_pretraining_chen.vision_transformer4k import  vit4k_xs

from torchinfo import summary 
import configs.config as config
from src.datamodules.dataset_4k import Train_data_surv_4k, Validation_data_surv_4k, Test_data_surv_4k
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score ,f1_score,roc_curve, auc,precision_recall_curve, average_precision_score
import wandb
import argparse, configparser
import os
import time
from src.utils import *
from src.losses import *
from einops import rearrange, repeat
from torchvision import transforms

from torchvision.models import vit_b_16 
from src.models.modules.vit_architecture import ViT

from torchvision.models.feature_extraction import create_feature_extractor


model256_path: str = '/data/PANDA/code/survival_prediction/Experiments/2023_07_19_18_09_03/checkpoints_300.pth'
model2k_path: str = '/data/PANDA/code/survival_prediction/Experiments/2023_08_17_00_17_17/checkpoints_126.pth'

# Hierarchical vision transformer
class VitToken(nn.Module):
    def __init__(self, model2k):
        super(VitToken, self).__init__()

        self.class_token = nn.Parameter(torch.randn(1, 1, 768), requires_grad=True)
        self.embedding_dropout = model2k.embedding_dropout
        self.transformer_encoder = model2k.transformer_encoder
        self.mlp_head = model2k.mlp_head


    def forward(self, x):
        class_token = self.class_token
        x = torch.cat((class_token.permute(1, 0, 2), x.unsqueeze(dim=1).permute(1, 0, 2)), dim=1)
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :] 
        x = self.mlp_head(x)
        
        return x
    


class VitSurvHier2k:
    def __init__(self, filename) :
        
        self.file_name = filename

        seed_everything(config.seed)

        self.model256 = vit_b_16() #pretrained_weights=model256_path)   
        self.model256.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)
        self.model2k = ViT(config) #pretrained_weights=model4k_path)

        self.model256.load_state_dict(torch.load(model256_path))

        self.feature_extractor_256 = create_feature_extractor(self.model256, return_nodes=['encoder.ln'])

        self.model = VitToken(self.model2k)

        self.train = self.training()


    #todo: resize patchsize fro 256 to 224    
    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):

        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)

        return img_new, w_256, h_256
    

    def training(self):
        if config.onlyEval == False:

            training_data = Train_data_surv_4k(config)
            train_loader = DataLoader(training_data,batch_size=config.batch_size,shuffle=True ) #, num_workers =self.config.num_workers)#, num_workers=config.num_workers)
            
            #file to save checkpoints
            file_name = os.path.join(self.file_name, "checkpoints")
            os.mkdir(file_name)


            model_256 = self.model256.to(config.device)
            model = self.model.to(config.device)



            for parameter in model_256.parameters():
                parameter.requires_grad = True

            for parameter in model.parameters():
                parameter.requires_grad = True

            
            optimizer = torch.optim.Adam(model.parameters(),
                                lr=config.lr, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), #, # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=config.weight_decay) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet
            
            
            loss_fn = nn.BCEWithLogitsLoss(pos_weight= weight_loss)

            loss_fn_val = nn.BCEWithLogitsLoss()

            
            train_loss, train_acc_overall = [],[]
            val_loss, val_acc_overall = [],[]

            print('Training:',model.training)

            for epoch in tqdm(range(config.epochs)):
                
                acc_batch, acc_batch_val = 0, 0
                epoch_loss, epoch_loss_val = [], []

                for idx,(image,label) in enumerate(train_loader):
                    
                    image = image.to(config.device,dtype = torch.float32)
                    label = label.to(config.device,dtype = torch.float32).unsqueeze(1)


                    batch_256, w_256, h_256 = self.prepare_img_tensor(image)                # 1. [1 x 3 x W x H].
                    batch_256 = batch_256.unfold(2, 224, 256).unfold(3, 224, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
                    batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)



                    features_cls256 = (self.feature_extractor_256(batch_256)['encoder.ln'][:, 0, :].detach().cpu())
                                          
                    features_cls256 = features_cls256.to(config.device, non_blocking=True)
                    predict_label = model(features_cls256) #[:, 0, :]    


                    loss = loss_fn(predict_label, label)
                    optimizer.zero_grad()
                    loss.backward()

                    
                    optimizer.step()
                    #step += 1
                    #print('step:', step)

                    epoch_loss.append(loss.item())

                    y_pred_class = torch.round(torch.sigmoid(predict_label.squeeze(1)))
                    acc_batch += (y_pred_class == label.squeeze(1)).sum().item()/len(predict_label)

                train_acc = (acc_batch / len(train_loader))
                train_acc_overall.append(train_acc)
                train_loss = np.array(epoch_loss).mean()

                print(f'train_loss: {train_loss:.4f}, train_Acc: {train_acc:.4f}')
                wandb.log({"epoch": epoch, "train accuracy": train_acc, "train_loss": train_loss})
                
                if epoch %  2 == 0:
                    if config.pretrain:
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth") 

                    else:
                        print('epoch:', epoch)
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth")


                    
                if config.validation == True:

                    if epoch % 1 == 0:

                        print('Validation Phase')

                        gt_val = []
                        predictions_val = []
                        pred_prob_val = []

 
                        val_data = Validation_data_surv_4k(config)
                        val_loader = DataLoader(val_data,batch_size=config.batch_size,shuffle=False ) #, num_workers =config.num_workers)
                        

                        with torch.no_grad():

                            for idx,(image,label) in enumerate(val_loader):

                                image = image.to(config.device,dtype = torch.float32)
                                label = label.to(config.device,dtype = torch.float32).unsqueeze(1)
                                
                                batch_256, w_256, h_256 = self.prepare_img_tensor(image)                # 1. [1 x 3 x W x H].
                                batch_256 = batch_256.unfold(2, 224, 256).unfold(3, 224, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
                                batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)



                                features_cls256 = (self.feature_extractor_256(batch_256)['encoder.ln'][:, 0, :].detach().cpu())
                                                    
                                features_cls256 = features_cls256.to(config.device, non_blocking=True)
                                predict_prob_val = model(features_cls256)

                                val_loss = loss_fn_val(predict_prob_val, label)

                                y_pred_class_val = torch.round(torch.sigmoid(predict_prob_val.squeeze(1)))
                                epoch_loss_val.append(val_loss.item())
                                acc_batch_val += (y_pred_class_val == label.squeeze(1)).sum().item()/len(predict_prob_val)


                                gt_val.append(label.squeeze(1).detach().cpu())
                                pred_prob_val.append(predict_prob_val.squeeze(1).detach().cpu())
                                predictions_val.append(y_pred_class_val.detach().cpu())


                            val_acc = (acc_batch_val / len(val_loader))
                            val_loss = np.array(epoch_loss_val).mean()
                            print(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

                            gt_val = ten_to_lst(gt_val)
                            pred_prob_val = ten_to_lst(pred_prob_val)
                            predictions_val = ten_to_lst(predictions_val)

                            cm_val_path = os.path.join(self.file_name,'configuration_matrix_val.png')

                            conf_matrix(gt_val,predictions_val,cm_val_path)
                            
                            auroc_val_path = os.path.join(self.file_name,'auroc_val.png')

                            f1, fpr, tpr, precision, recall , roc_auc , auprc = other_scores(gt_val,pred_prob_val,predictions_val,auroc_val_path)


                            wandb.log({"epoch": epoch, "train accuracy": train_acc, "train_loss": train_loss,
                                        "val_acc": val_acc, "validation loss": val_loss, 
                                        "Confusion matrix val": wandb.Image(cm_val_path),
                                        "AUROC_val": wandb.Image(auroc_val_path)})



            print('training and validating done')
                        
def plot_auroc(file_name, fpr, tpr, roc_auc):
    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(file_name)

def conf_matrix(gt,predictions,file_name):
    print('Confusion matrix:', confusion_matrix(gt , predictions))
    disp = ConfusionMatrixDisplay.from_predictions(gt, predictions, normalize= 'true', cmap = 'Blues') 
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(file_name)


def other_scores(gt, pred_prob, predictions, file_name):
    f1 = f1_score(gt, predictions)
    print('F1 Score:', f1)

    fpr, tpr, threshold = roc_curve(gt, pred_prob)
    roc_auc = auc(fpr, tpr)

    print('Area Under Curve:', roc_auc)
    plot_auroc(file_name, fpr, tpr, roc_auc)

    precision, recall, thresholds = precision_recall_curve(gt, predictions)
    print('Precision:', precision)
    print('Recall:', recall)

    auprc = average_precision_score(gt, predictions)

    return f1, fpr, tpr, precision, recall , roc_auc , auprc