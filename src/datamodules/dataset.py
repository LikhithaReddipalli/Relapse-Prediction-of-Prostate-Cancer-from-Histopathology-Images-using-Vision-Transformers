import os
import sys
#sys.path.append('/home/likhitha/prostate_cancer_vit/code'
import platform

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F

from torch.utils.data import DataLoader, Dataset

import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rescale

import configs.config as config

from src.utils import *
from db_helper import *



#TODO: write one dataloader to load data - pass arugument as train, val, test 
#TODO: add path to db - may be not always change the path in dataloaders

# UKE dataloader
class TrainDataSurv(Dataset):
    def __init__(self,config):

        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.mean = config.mean
        self.std  = config.std

        self.manual_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])^123

        self.data_aug_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(config.resize),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor() ])
                                                            #transforms.Normalize(mean=config.mean, 
                                                                                #std=config.std) 
            
        df  = get_db_table('uke.experiments.clas_60_months','survival_prediction\db_lr.sqlite')

        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        df = df[11:50]
                
        self.images = df.loc[df['split'] == 'train', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'train', 'has_event_before_60_months'].tolist()


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)

        if config.data_augmentation == True :
            image = self.data_aug_transforms(image)
        else:
            image = self.manual_transforms(image)

        #show_patch_image(image,224,16)

        label = self.labels[item]  
        
        return image, label

# Validation class

class ValidationDataSurv(Dataset):

    def __init__(self,config):
        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.manual_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(config.resize),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor() ])
                                                            #transforms.Normalize(mean=config.mean, 
                                                                                #std=config.std)


        df  = get_db_table('uke.experiments.clas_60_months','survival_prediction\db_lr.sqlite')

        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        df = df[11:50]
                
        self.images = df.loc[df['split'] == 'val', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'val', 'has_event_before_60_months'].tolist()


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)
        image = self.manual_transforms(image)
        label = self.labels[item]  
        
        return image, label



# Test class
class TestDataSurv(Dataset):

    def __init__(self,config):
        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.manual_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(config.resize),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor() ])
                                                            #transforms.Normalize(mean=config.mean, 
                                                                                #std=config.std)])

        df  = get_db_table('uke.experiments.clas_60_months','survival_prediction\db_lr.sqlite')
    
        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        self.images = df.loc[df['split'] == 'test', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'test', 'has_event_before_60_months'].tolist()

                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)
        image = self.manual_transforms(image)
        label = self.labels[item]  
        
        return image, label
    

def find_classes(directory):
    df = pd.read_csv(directory)
    df_labels = df[['pred_event_before_2y_667']]
    classes = df_labels.pred_event_before_2y_667.unique()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def image_rescale(image):
    image = np.asarray(image)
    image = rescale(image,config.rescale_factor,anti_aliasing=None)
    return image

def random_rotate(image):
    angle = random.choice(config.degrees)
    return F.rotate(image,angle)


def join_string(values, join_char):
    return join_char.join(values)



class TrainDataPatch(Dataset):

    def __init__(self,config):

        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.manual_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        self.data_aug_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(224),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=config.mean, 
                                                                                std=config.std)  ])

        df  = get_db_table('uke.experiments.clas_60_months','/data/PANDA/code/survival_prediction/db_lr.sqlite')

        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        #df = df[0:20]
                
        self.images = df.loc[df['split'] == 'train', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'train', 'has_event_before_60_months'].tolist()
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)
        #im = np.array(image)

        patches = []
        for i in range(0, config.img_size, config.patch_size):
            for j in range(0, config.img_size, config.patch_size):
                patch = image.crop((i, j, i + config.patch_size, j + config.patch_size))
                patch = F.resize(patch, (224, 224))
                patch = F.to_tensor(patch)
                patches.append(patch)

        # Stack the patches into a stacked_image
        stacked_image = torch.stack(patches, dim=0)
        #show_patch_image(image)

        label = self.labels[item]  
        return stacked_image, label

class ValidationDataPatch(Dataset):

    def __init__(self,config):

        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.manual_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        self.data_aug_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(224),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor() ])
                                                            #transforms.Normalize(mean=config.mean, 
                                                                                #std=config.std) 

        #'/data/PANDA/code/survival_prediction/db_lr.sqlite'
        df  = get_db_table('uke.experiments.clas_60_months','/data/PANDA/code/survival_prediction/db_lr.sqlite')

        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        #df = df[0:10]
                
        self.images = df.loc[df['split'] == 'val', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'val', 'has_event_before_60_months'].tolist()
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)#
        #im = np.array(image)
        patches = []
        for i in range(0, config.img_size, config.patch_size):
            for j in range(0, config.img_size, config.patch_size):
                patch = image.crop((i, j, i + config.patch_size, j + config.patch_size))
                patch = F.resize(patch, (224, 224))
                patch = F.to_tensor(patch)
                patches.append(patch)

        # Stack the patches into a stacked_image
        stacked_image = torch.stack(patches, dim=0)
        #show_patch_image(image)

        label = self.labels[item]  
        return stacked_image, label


class TestDataPatch(Dataset):
    def __init__(self,config):

        self.data_dir = config.data_dir_surv
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.images = []  
        self.labels = []
        self.manual_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        self.data_aug_transforms = transforms.Compose([#transforms.ToPILImage(),
                                                            transforms.Resize(224),
                                                            #transforms.CenterCrop(224),
                                                            transforms.ToTensor() ])
                                                            #transforms.Normalize(mean=config.mean, 
                                                                                #std=config.std) 

        df  = get_db_table('uke.experiments.clas_60_months','/data/PANDA/code/survival_prediction/db_lr.sqlite')

        df['filepath'] = df['filepath'].str.replace('_true.tif', '.png')
        df['filepath'] = self.data_dir + '/' + df['filepath']
        df['filepath'] = df['filepath'].str.replace('/Images', '')

        #df = df[0:16]
                
        self.images = df.loc[df['split'] == 'test', 'filepath'].tolist()
        self.labels = df.loc[df['split'] == 'test', 'has_event_before_60_months'].tolist()


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):

        img_path = self.images[item]
        image = Image.open(img_path)
        #im = np.array(image)

        patches = []
        for i in range(0, config.img_size, config.patch_size):
            for j in range(0, config.img_size, config.patch_size):
                patch = image.crop((i, j, i + config.patch_size, j + config.patch_size))
                patch = F.resize(patch, (224, 224))
                patch = F.to_tensor(patch)
                patches.append(patch)

        # Stack the patches into a stacked_image
        stacked_image = torch.stack(patches, dim=0)

        label = self.labels[item]  
    
        return stacked_image, label, img_path
    


if __name__== '__main__':
    base_path = '/data'
    train_dir = base_path + '/training_final.csv'
    val_dir = base_path + '/validation_final.csv'
    test_dir = base_path + '/test_final.csv'
    data_dir = 'data\png_original_size_squares'
    img_size = config.img_size
    patch_size = config.patch_size

    training_data = Train_data_surv(config)
    train_loader = DataLoader(training_data,batch_size=2,shuffle=False)
    val_data = Validation_data_surv(config)
    val_loader = DataLoader(val_data,batch_size=2,shuffle=False)
    testing_data = Test_data_surv(config)
    test_loader = DataLoader(testing_data,batch_size=2,shuffle=False)

    print('training data')
    for idx,(image,label) in enumerate(train_loader):
        print(f'idx:{idx}')
        print(f'idx:{idx}, class:{label}, shape: {image.shape}')

    print('validation data')
    for idx,(image,label) in enumerate(val_loader):
        print(f'idx:{idx}')
        print(f'idx:{idx}, class:{label}, shape: {image.shape}')

    print('testing data')
    for idx,(image,label) in enumerate(test_loader):
        print(f'idx:{idx}')
        print(f'idx:{idx}, class:{label}, shape: {image.shape}')