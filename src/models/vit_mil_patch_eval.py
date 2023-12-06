import os
import sys
sys.path.append('C:\\Users\\likhi\\Desktop\\Thesis\\pc\\pc_code\\gleason_prediction')
import wandb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary 
from torch.utils.data import DataLoader
import torchvision

from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
from torchvision.models import vit_b_16 , vit_l_16


import configs.config as config

import matplotlib.pyplot as plt
from tqdm.rich import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score , f1_score,roc_curve, auc,precision_recall_curve, average_precision_score


from src.utils import *
from src.losses import *
from src.datamodules.dataset import TrainDataPatch, ValidationDataPatch, TestDataPatch
from src.models.modules.vit_patch_model import CustomModelViT, CustomModelViT_Test


# Evaluation of UKE dataset relapse prediction using vision transformers with multiple instance learning
class VitSurvPatchEval:
    def __init__(self, filename) :
        

        self.file_name = filename
        self.model = CustomModelViT_Test(config)
        self.test = self.testing()


    def testing(self):
        model = self.model.to(config.device)

        loss_fn_test = nn.BCEWithLogitsLoss()


        gt_test = []
        predictions_test = []
        pred_prob_test = []


        gt_test1 = []
        predictions_test1 = []
        pred_prob_test1 = []
        filenames_test = []


        test_data = TestDataPatch(config)
        test_loader = DataLoader(test_data,batch_size=config.batch_size,shuffle=False, pin_memory= True)

        with torch.no_grad():


            acc_batch, acc_batch_test = 0, 0
            epoch_loss, epoch_loss_test = [], []

            with tqdm(range(len(test_loader))) as pbar:


                for idx,(stacked_image,label,img_path) in enumerate(test_loader):


                    stacked_image = stacked_image.to(config.device,dtype = torch.float32)
                    label = label.to(config.device,dtype = torch.float32) #.unsqueeze(1)

                    image = stacked_image.view(stacked_image.size()[0] * config.num_patches , config.num_channels, config.resize, config.resize)
                    
                    predict_prob_test = model(image)
                    predict_prob_test = predict_prob_test.squeeze()
                    test_loss = loss_fn_test(predict_prob_test, label)

                    y_pred_class_test = torch.round(torch.sigmoid(predict_prob_test))    #.squeeze(1)))
                    epoch_loss_test.append(test_loss.item())
                    acc_batch_test += (y_pred_class_test == label).sum().item()/len(predict_prob_test)


                    gt_test.append(label.detach().cpu())
                    pred_prob_test.append(predict_prob_test.detach().cpu())
                    predictions_test.append(y_pred_class_test.detach().cpu())

                    gt_test1.append(label.detach().cpu())
                    pred_prob_test1.append(predict_prob_test.detach().cpu())
                    predictions_test1.append(y_pred_class_test.detach().cpu())
                    filenames_test.extend(img_path)


                    labels_array = np.concatenate(gt_test, axis=0)
                    logits_array = np.concatenate(pred_prob_test, axis=0)
                    predictions_array = np.concatenate(predictions_test, axis=0)
                    filenames_array = np.array(filenames_test)

            np.savez(os.path.join(self.file_name,'results_test.npz'), labels = labels_array, logits=logits_array, predictions=predictions_array, filenames=filenames_array)


            test_acc = (acc_batch_test / len(test_loader))
            test_loss = np.array(epoch_loss_test).mean()
            print(f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}')

            gt_test = ten_to_lst(gt_test)
            pred_prob_test = ten_to_lst(pred_prob_test)
            predictions_test = ten_to_lst(predictions_test)

            cm_test_path = os.path.join(self.file_name,'configuration_matrix_test.png')

            conf_matrix(gt_test,predictions_test,cm_test_path)
            
            auroc_test_path = os.path.join(self.file_name,'auroc_test.png')

            f1, fpr, tpr, precision, recall , roc_auc , auprc = other_scores(gt_test,pred_prob_test,predictions_test,auroc_test_path)


            wandb.log({
                        "test accuracy": test_acc, "test loss": test_loss, 
                        "Confusion matrix val": wandb.Image(cm_test_path),
                        "AUROC_val": wandb.Image(auroc_test_path)})



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

