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

# Pretrain model evaluation

class PandaEval:

    def __init__(self, filename, config):
        
        self.file_name = filename
        self.config = config
        #self.model = ViT(self.config)
        self.experiment_dfs = load_experiment_dfs(pandaconfig) 
        self.coords_df = load_coords_df(self.experiment_dfs) # here only traindata is being loaded
        #self.weights = losses(self.coords_df)       
        self.validating = self.valid()


    def valid(self):

        model = vit_b_16()
        model.heads = nn.Linear(in_features=self.config.embedding_dim, out_features=self.config.num_classes).to(self.config.device)
        model = model.to(self.config.device)

        print('Loading checkpoints')
        model.load_state_dict(torch.load(self.config.checkpoint))

        model.eval()
        model.requires_grad_(False)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_fn_val = nn.BCEWithLogitsLoss()

        if self.config.validation == True:



            print('Validation Phase')

            gt_val = []
            predictions_val = []
            pred_prob_val = []
            acc_batch, acc_batch_val = 0, 0
            epoch_loss, epoch_loss_val = [], []

            val_data = Val_data_panda(pandaconfig,self.coords_df)
            val_loader = DataLoader(val_data,batch_size=self.config.batch_size,shuffle=False,num_workers =self.config.num_workers)

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


                wandb.log({
                            "val_acc": val_acc, "validation loss": val_loss, 
                            "Confusion matrix val": wandb.Image(cm_val_path),
                            "AUROC_val": wandb.Image(auroc_val_path)})





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
