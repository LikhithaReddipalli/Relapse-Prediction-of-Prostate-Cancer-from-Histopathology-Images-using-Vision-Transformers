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
model_path: str = '/data/PANDA/code/survival_prediction/Experiments/2023_09_20_15_35_58/checkpoints_14.pth'

# Evaluation of hierarchical vision transformer
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
    


class VitSurvHier2kEval:
    def __init__(self, filename) :
        
        self.file_name = filename

        seed_everything(config.seed)

        self.model256 = vit_b_16() #pretrained_weights=model256_path)   
        self.model256.heads = nn.Linear(in_features=config.embedding_dim, out_features=config.num_classes).to(config.device)
        self.model2k = ViT(config) #pretrained_weights=model4k_path)

        self.model256.load_state_dict(torch.load(model256_path , map_location='cuda:0'))
        self.feature_extractor_256 = create_feature_extractor(self.model256, return_nodes=['encoder.ln'])

        self.model = VitToken(self.model2k)
        self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        self.train = self.testing()


    #todo: resize patchsize fro 256 to 224    
    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):

        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)

        return img_new, w_256, h_256
    

    def testing(self):
                
        if config.test == True:

            print('Testing Phase')           

            testing_data = Test_data_surv_4k(config)
            test_loader = DataLoader(testing_data,batch_size=config.batch_size,shuffle=False)
            

            model_256 = self.model256.to(config.device)
            model = self.model.to(config.device)

            model.eval()
            model_256.eval()
            model_256.requires_grad_(False)
            model.requires_grad_(False)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            
            gt = []
            predictions = []
            pred_prob = []
            accuracy = []
            
            gt_test1 = []
            predictions_test1 = []
            pred_prob_test1 = []
            filenames_test = []

            with tqdm(range(len(test_loader))) as pbar:

                for idx,(image,label,img_path) in enumerate(test_loader):
                    
                    image = image.to(config.device,dtype = torch.float32)
                    label = label.to(config.device,dtype = torch.float32).unsqueeze(1)


                    batch_256, w_256, h_256 = self.prepare_img_tensor(image)                # 1. [1 x 3 x W x H].
                    batch_256 = batch_256.unfold(2, 224, 256).unfold(3, 224, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
                    batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)



                    features_cls256 = (self.feature_extractor_256(batch_256)['encoder.ln'][:, 0, :].detach().cpu())
                                          
                    features_cls256 = features_cls256.to(config.device, non_blocking=True)
                    predict_label = self.model(features_cls256)
                                        
                    loss = loss_fn(predict_label, label)

                    y_pred_class = torch.round(torch.sigmoid(predict_label.squeeze(1)))

                    accuracy.append(accuracy_score(y_pred_class.detach().cpu(), label.squeeze(1).detach().cpu()))

                    pred_prob.append(predict_label.squeeze(1).detach().cpu())

                    predictions.append(y_pred_class.detach().cpu())

                    gt.append(label.squeeze(1).detach().cpu())
                                        
                    
                    gt_test1.append(label.detach().cpu())
                    pred_prob_test1.append(predict_label.detach().cpu())
                    predictions_test1.append(y_pred_class.detach().cpu())
                    filenames_test.extend(img_path)


                    labels_array = np.concatenate(gt, axis=0)
                    logits_array = np.concatenate(pred_prob, axis=0)
                    predictions_array = np.concatenate(predictions, axis=0)
                    filenames_array = np.array(filenames_test)

                np.savez(os.path.join(self.file_name,'results_test.npz'), labels = labels_array, logits=logits_array, predictions=predictions_array, filenames=filenames_array)

                        
            gt = ten_to_lst(gt)
            predictions = ten_to_lst(predictions)
            pred_prob = ten_to_lst(pred_prob)

            accuracy = sum(accuracy) / len(accuracy)
            
            print('Accuracy of the model:', accuracy)
            
            
            
            

            cm_test_path = os.path.join(self.file_name,'configuration_matrix.png')

            conf_matrix(gt,predictions,cm_test_path)

            auroc_test_path = os.path.join(self.file_name,'auroc.png')

            f1, fpr, tpr, precision, recall , roc_auc , auprc = other_scores(gt,pred_prob,predictions,auroc_test_path)

            wandb_cm = wandb.plot.confusion_matrix(probs=None, y_true= gt, preds = predictions, class_names =[ 'class : 0' ,'class: 1'])

            wandb.log({"Accuracy" : accuracy , "Confusion matrix" : wandb_cm , "Precision": precision, "recall": recall, "f1 score": f1 , "fpr": fpr, "tpr": tpr, "AUROC":roc_auc, "AUPRC": auprc , 
                    "Confusion matrix": wandb.Image(cm_test_path),
                    "AUROC": wandb.Image(auroc_test_path)
                    })


            print('Testing done')
                        
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



