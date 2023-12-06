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
from src.models.modules.vit_patch_model import CustomModelViT


# UKE dataset relapse prediction using vision transformers with multiple instance learning

class VitSurvPatch:
    def __init__(self, filename) :
        

        self.file_name = filename
        self.model = CustomModelViT(config)
        self.train = self.training()
        

    def training(self):

        if config.onlyEval == False:

            training_data = TrainDataPatch(config)
            train_loader = DataLoader(training_data,batch_size=config.batch_size,shuffle=True, pin_memory= True)

            #file to save checkpoints
            file_name = os.path.join(self.file_name, "checkpoints")
            os.mkdir(file_name)

            #pretrained_model = efficientnet_b0(weights= EfficientNet_B0_Weights.DEFAULT).to(config.device)

            #model = CustomModel(pretrained_model)

            model = self.model.to(config.device)

            optimizer = torch.optim.Adam(model.parameters(),
                                lr=config.lr,
                                betas= config.betas)
                                #weight_decay= config.weight_decay 
            

            loss_fn = nn.BCEWithLogitsLoss(pos_weight = weight_loss)
            loss_fn_val = nn.BCEWithLogitsLoss()

            
            train_loss, train_acc_overall = [],[]
            val_loss, val_acc_overall = [],[]

            print('Training Phase: ',model.training)

            for epoch in tqdm(range(config.epochs)):
                
                acc_batch, acc_batch_val = 0, 0
                epoch_loss, epoch_loss_val = [], []
                for idx,(stacked_image,label) in enumerate(train_loader):


                    stacked_image = stacked_image.to(config.device,dtype = torch.float32)
                    label = label.to(config.device,dtype = torch.float32) #.unsqueeze(1)

                    #image = stacked_image.contiguous().view(-1, *stacked_image.shape[2:])

                    flat_output = stacked_image.view(stacked_image.size()[0] * config.num_patches , config.num_channels, config.resize, config.resize)

                    predict_label =model(flat_output)


                    predict_label = predict_label.squeeze()

                    loss = loss_fn(predict_label, label)
                    optimizer.zero_grad()
                    loss.backward()
                    
                    optimizer.step()

                    epoch_loss.append(loss.item())

                    y_pred_class = torch.round(torch.sigmoid(predict_label)) #.squeeze(1)))
                    acc_batch += (y_pred_class == label).sum().item()/len(predict_label)

                train_acc = (acc_batch / len(train_loader))
                train_acc_overall.append(train_acc)
                train_loss = np.array(epoch_loss).mean()
                print(f'train_loss: {train_loss:.4f}, train_Acc: {train_acc:.4f}')
                wandb.log({"epoch": epoch, "train accuracy": train_acc, "train_loss": train_loss})
                
                if epoch % 1 == 0:
                    if config.pretrain:
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth") 

                    else:
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth")

                torch.save(model, f"{file_name}_model.pth")

                if config.validation == True:

                    if epoch % 1 == 0:

                        print('Validation Phase')

                        gt_val = []
                        predictions_val = []
                        pred_prob_val = []

                        val_data = ValidationDataPatch(config)
                        val_loader = DataLoader(val_data,batch_size=config.batch_size,shuffle=False, pin_memory= True)

                        with torch.no_grad():

                            for idx,(stacked_image,label) in enumerate(val_loader):


                                stacked_image = stacked_image.to(config.device,dtype = torch.float32)
                                label = label.to(config.device,dtype = torch.float32) #.unsqueeze(1)

                                image = stacked_image.view(stacked_image.size()[0] * config.num_patches , config.num_channels, config.resize, config.resize)
                                
                                predict_prob_val = model(image)
                                predict_prob_val = predict_prob_val.squeeze()
                                val_loss = loss_fn_val(predict_prob_val, label)

                                y_pred_class_val = torch.round(torch.sigmoid(predict_prob_val)) #.squeeze(1)))
                                epoch_loss_val.append(val_loss.item())
                                acc_batch_val += (y_pred_class_val == label).sum().item()/len(predict_prob_val)


                                gt_val.append(label.detach().cpu())
                                pred_prob_val.append(predict_prob_val.detach().cpu())
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

                            

                            f1, fpr, tpr, precision, recall , roc_auc , auprc , roc_data = other_scores(gt_val,pred_prob_val,predictions_val,auroc_val_path)


                            wandb.log({"epoch": epoch, "train accuracy": train_acc, "train_loss": train_loss,
                                        "validation accuracy": val_acc, "validation loss": val_loss, 
                                        "Confusion matrix val": wandb.Image(cm_val_path),
                                        "AUROC_val": wandb.Image(auroc_val_path)
                                        })

                    if epoch % 10 == 0:


                        print('Testing')


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


                            acc_batch_test = 0
                            epoch_loss_test = []
                            loss_fn_test =nn.BCEWithLogitsLoss()

                            #with tqdm(range(len(test_loader))) as pbar:


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

                            f1, fpr, tpr, precision, recall , roc_auc , auprc, roc_data = other_scores(gt_test,pred_prob_test,predictions_test,auroc_test_path)


                            wandb.log({
                                        "test accuracy": test_acc, "test loss": test_loss, 
                                        "Confusion matrix test": wandb.Image(cm_test_path),
                                        "AUROC_test": wandb.Image(auroc_test_path),
                                        })
                            
                            #"roc_test":wandb.plot.roc_curve(gt_test, pred_prob_test)




            

        
            
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
    disp = ConfusionMatrixDisplay.from_predictions(gt, predictions, normalize= 'true', cmap = 'Blues', im_kw= dict(vmin=0,vmax=1)) 
    #disp.im_.set_clim(0, 1)
    #disp.plot(cmap=plt.cm.Blues)
    plt.savefig(file_name)


def other_scores(gt, pred_prob, predictions, file_name):
    f1 = f1_score(gt, predictions)
    print('F1 Score:', f1)

    fpr, tpr, threshold = roc_curve(gt, pred_prob)
    roc_auc = auc(fpr, tpr)

    roc_data = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": threshold.tolist()}

    print('Area Under Curve:', roc_auc)
    plot_auroc(file_name, fpr, tpr, roc_auc)

    precision, recall, thresholds = precision_recall_curve(gt, predictions)
    print('Precision:', precision)
    print('Recall:', recall)

    auprc = average_precision_score(gt, predictions)



    return f1, fpr, tpr, precision, recall , roc_auc , auprc, roc_data




    

    

