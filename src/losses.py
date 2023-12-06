from db_helper import *
import torch.nn as nn
import torch
import configs.config as config

df = get_db_table('uke.experiments.clas_60_months','/data/PANDA/code/survival_prediction/db_lr.sqlite') 

len_training = len(df[df['split'] == 'train'].value_counts())
neg_val = df[df['split'] == 'train']['has_event_before_60_months'].value_counts()[0]
pos_val = df[df['split'] == 'train']['has_event_before_60_months'].value_counts()[1]
weight_loss = torch.tensor(neg_val/pos_val)        

