import torch
from torch.utils.data import RandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from src.datamodules.dataset_pretrain import *
from tqdm.rich import tqdm
import numpy as np
import os
from src.models.modules.vit_architecture import ViT
from src.utils import *
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score ,f1_score,roc_curve, auc,precision_recall_curve, average_precision_score
import configs.panda as pandaconfig
from src.datamodules.dataset_pretrain import *

from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
from torchvision.models import vit_b_16 , vit_l_16

from torchvision.models import efficientnet_b0 , EfficientNet_B0_Weights


# Panda pretraining for CNN

class PretrainPandaCnn:
    def __init__(self, filename, config):
        
        self.file_name = filename
        self.config = config
        #self.model = CustomModelViT_test(self.config)
        
        self.experiment_dfs = load_experiment_dfs(pandaconfig) 
        self.coords_df = load_coords_df(self.experiment_dfs) # here only traindata is being loaded
        self.weights = losses(self.coords_df)       
        self.train = self.training()

    def training(self):
        if self.config.onlyEval == False:

            training_data = TrainDataPanda(pandaconfig,self.coords_df)
            train_loader = DataLoader(training_data,batch_size=self.config.batch_size,shuffle=False,sampler =RandomSampler(training_data, replacement=False, num_samples= 256000) ) #, num_workers =self.config.num_workers)
            

            self.model = efficientnet_b0(weights= EfficientNet_B0_Weights.DEFAULT)

            self.model.classifier = nn.Sequential(
                                                nn.Dropout(p=0.2, inplace=True),
                                                nn.Linear(in_features=1280, out_features=self.config.num_classes, bias=True)
                                                )


            #file to save checkpoints
            file_name = os.path.join(self.file_name, "checkpoints")
            os.mkdir(file_name)

            model = self.model.to(self.config.device) 

            for parameter in model.parameters():
                parameter.requires_grad = True

            optimizer = torch.optim.Adam(model.parameters(),
                                lr=self.config.lr, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=self.config.betas) # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                #weight_decay=self.config.weight_decay) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet
            
            
            #loss_fn = nn.BCEWithLogitsLoss(pos_weight= weight_loss)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight= self.weights) #pos_weight= self.weights
            loss_fn_val = nn.BCEWithLogitsLoss()

            train_loss, train_acc_overall = [],[]
            val_loss, val_acc_overall = [],[]

            step = 0

            print('Training:',model.training)
            for epoch in tqdm(range(self.config.epochs)):
                
                acc_batch, acc_batch_val = 0, 0
                epoch_loss, epoch_loss_val = [], []

                for idx,(image,label,dict) in enumerate(train_loader):
                    
                    image = image.to(self.config.device,dtype = torch.float32)
                    label = label.to(self.config.device,dtype = torch.float32).unsqueeze(1)

                    predict_label = model(image)

                    loss = loss_fn(predict_label, label)
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    step += 1
                    print('step:', step)

                    epoch_loss.append(loss.item())

                    y_pred_class = torch.round(torch.sigmoid(predict_label.squeeze(1)))
                    acc_batch += (y_pred_class == label.squeeze(1)).sum().item()/len(predict_label)

                train_acc = (acc_batch / len(train_loader))
                train_acc_overall.append(train_acc)
                train_loss = np.array(epoch_loss).mean()

                print(f'train_loss: {train_loss:.4f}, train_Acc: {train_acc:.4f}')
                wandb.log({"epoch": epoch, "train accuracy": train_acc, "train_loss": train_loss})
                
                if epoch %  2 == 0:
                    if self.config.pretrain:
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth") 

                    else:
                        print('epoch:', epoch)
                        torch.save(model.state_dict(),f"{file_name}_{epoch}.pth")


                    
                if self.config.validation == True:

                    if epoch % 1 == 0:

                        print('Validation Phase')

                        gt_val = []
                        predictions_val = []
                        pred_prob_val = []

                        val_data = ValDataPSanda(pandaconfig,self.coords_df)
                        val_loader = DataLoader(val_data,batch_size=self.config.batch_size,shuffle=False ) #, num_workers =self.config.num_workers)
                        

                        with torch.no_grad():

                            for idx,(image,label,dict) in enumerate(val_loader):

                                image = image.to(self.config.device,dtype = torch.float32)
                                label = label.to(self.config.device,dtype = torch.float32).unsqueeze(1)
                                
                                predict_prob_val = model(image)
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


            print('------ Evaluation ------')
                
            testing_data = TestDataPanda(pandaconfig,self.coords_df)
            test_loader = DataLoader(testing_data,batch_size=self.config.batch_size,shuffle=False, num_workers =self.config.num_workers)
            
            model.eval()
            model.requires_grad_(False)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            
            gt = []
            predictions = []
            pred_prob = []
            accuracy = []

            with tqdm(range(len(test_loader))) as pbar:

                for idx,(image,label,dict) in enumerate(test_loader):
                    
                    image = image.to(self.config.device,dtype = torch.float32)
                    label = label.to(self.config.device,dtype = torch.float32).unsqueeze(1)
                    
                    predict_label = model(image)
                                        
                    loss = loss_fn(predict_label, label)

                    y_pred_class = torch.round(torch.sigmoid(predict_label.squeeze(1)))

                    accuracy.append(accuracy_score(y_pred_class.detach().cpu(), label.squeeze(1).detach().cpu()))

                    pred_prob.append(predict_label.squeeze(1).detach().cpu())

                    predictions.append(y_pred_class.detach().cpu())

                    gt.append(label.squeeze(1).detach().cpu())
                
            gt = ten_to_lst(gt)
            predictions = ten_to_lst(predictions)
            pred_prob = ten_to_lst(pred_prob)

            accuracy = sum(accuracy) / len(accuracy)
            
            print('Accuracy of the model:', accuracy) 

            cm_test_path = os.path.join(self.file_name,'configuration_matrix.png')

            conf_matrix(gt,predictions,cm_test_path)

            auroc_test_path = os.path.join(file_name,'auroc.png')

            f1, fpr, tpr, precision, recall , roc_auc , auprc = other_scores(gt,pred_prob,predictions,auroc_test_path)

            wandb_cm = wandb.plot.confusion_matrix(probs=None, y_true= gt, preds = predictions, class_names =[ 'class : 0' ,'class: 1'])

            wandb.log({"Accuracy" : accuracy , "Confusion matrix" : wandb_cm , "Precision": precision, "recall": recall, "f1 score": f1 , "fpr": fpr, "tpr": tpr, "AUROC":roc_auc, "AUPRC": auprc , 
                    "Confusion matrix": wandb.Image(cm_test_path),
                    "AUROC": wandb.Image(auroc_test_path)
                    })
  
        

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






            
            
        