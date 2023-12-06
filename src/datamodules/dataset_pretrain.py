import os
import sys
sys.path.append('C:\\Users\\likhi\\Desktop\\Thesis\\pc\\pc_code\\survival_prediction')
import torch
from db_helper import *
from torch.utils.data import DataLoader, Dataset
import configs.panda as pandaconfig
from db_helper import *
from src.patch_preparer.patch_mask_preparer import PatchMaskPreparer
from src.transforms.image_transforms import ImageTransforms
from src.transforms.label_transforms import LabelTransformsClas
from src.datamodules.io_loader.openslide_patchloader import OpenslidePatchLoader
import random


# Panda dataloader
class TrainDataPanda(Dataset):

    def __init__(self,config,coords_df):

        self.config = config

        self.coords_df = coords_df.loc[lambda df_: df_["split"] == config.stage]
        self.image_transforms_train = ImageTransforms(config, augment=True, normalize=True)
        self.label_transforms_train = LabelTransformsClas()

        unused_coords = self.coords_df.loc[lambda df_: df_["split"] == config.stage].loc[
            lambda df_: ~df_.index.isin(self.coords_df.index)
        ]

        # remove unused from coords df stage
        self.coords_df = self.coords_df.loc[lambda df_: ~df_.index.isin(unused_coords.index)]    
        #print("coord",len(self.coords_df[:15]))

        self.label_dfs = {}
        
        self.label_dfs[config.stage] = self.coords_df['label']
        #print(self.label_dfs)
        
        
    def __len__(self):
        return len(self.coords_df)
    
    def __getitem__(self, item):

        curr_patch_info = self.coords_df.iloc[item]

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.config.image_base_dir, curr_patch_info["filename"]), self.config.patch_size
        )

        patch = patch_loader.get_patch(curr_patch_info["row"], curr_patch_info["col"])
        label = curr_patch_info["label"].astype("long")

        if self.image_transforms_train:
            patch = self.image_transforms_train(patch)

        if self.label_transforms_train:
            label = self.label_transforms_train(label)

        return patch, label, {"patch_info": curr_patch_info.to_dict()}


class ValDataPanda(Dataset):

    def __init__(self,config,coords_df):

        self.config = config

        self.coords_df = coords_df.loc[lambda df_: df_["split"] == config.stage]
        self.coords_df = coords_df.loc[lambda df_: df_["split"] == 'val']
        self.image_transforms_train = ImageTransforms(config, augment=False, normalize= True)
        self.label_transforms_train = LabelTransformsClas()

        unused_coords = self.coords_df.loc[lambda df_: df_["split"] == config.stage].loc[
            lambda df_: ~df_.index.isin(self.coords_df.index)
        ]

        # remove unused from coords df stage
        self.coords_df = self.coords_df.loc[lambda df_: ~df_.index.isin(unused_coords.index)]

        count =  25600 #int(len(self.coords_df) *0.01)

        df_label_0 = self.coords_df[self.coords_df['label'] == 0]
        df_label_1 = self.coords_df[self.coords_df['label'] == 1]

        sample_label_0 = df_label_0.sample(n= int(count/2), random_state=42)
        sample_label_1 = df_label_1.sample(n= int(count/2), random_state=42)

        # Concatenate the sampled DataFrames
        sampled_df = pd.concat([sample_label_0, sample_label_1])
        sampled_df = sampled_df.sample(frac=1, random_state=42)
        self.coords_df = sampled_df
    
        self.label_dfs = {}

        # get label_df from patch level for current dataset
        self.label_dfs[config.stage] = self.coords_df['label']
        #print(self.label_dfs)
        
        

    def __len__(self):
        return len(self.coords_df)
    
    def __getitem__(self, item):

        curr_patch_info = self.coords_df.iloc[item]

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.config.image_base_dir, curr_patch_info["filename"]), self.config.patch_size
        )

        patch = patch_loader.get_patch(curr_patch_info["row"], curr_patch_info["col"])
        label = curr_patch_info["label"].astype("long")

        if self.image_transforms_train:
            patch = self.image_transforms_train(patch)

        if self.label_transforms_train:
            label = self.label_transforms_train(label)

        return patch, label, {"patch_info": curr_patch_info.to_dict()}

class TestDataPanda(Dataset):

    def __init__(self,config,coords_df):

        self.config = config

        self.coords_df = coords_df.loc[lambda df_: df_["split"] == 'test']
        self.image_transforms_train = ImageTransforms(config, augment=False, normalize= True)
        self.label_transforms_train = LabelTransformsClas()

        unused_coords = self.coords_df.loc[lambda df_: df_["split"] == config.stage].loc[
            lambda df_: ~df_.index.isin(self.coords_df.index)
        ]

        # remove unused from coords df stage
        self.coords_df = self.coords_df.loc[lambda df_: ~df_.index.isin(unused_coords.index)]

        self.label_dfs = {}

        # get label_df from patch level for current dataset
        self.label_dfs[config.stage] = self.coords_df['label']
        #print(self.label_dfs)
        
        

    def __len__(self):
        return len(self.coords_df)
    
    def __getitem__(self, item):

        curr_patch_info = self.coords_df.iloc[item]

        patch_loader = OpenslidePatchLoader(
            os.path.join(self.config.image_base_dir, curr_patch_info["filename"]), self.config.patch_size
        )

        patch = patch_loader.get_patch(curr_patch_info["row"], curr_patch_info["col"])
        label = curr_patch_info["label"].astype("long")

        if self.image_transforms_train:
            patch = self.image_transforms_train(patch)

        if self.label_transforms_train:
            label = self.label_transforms_train(label)

        return patch, label, {"patch_info": curr_patch_info.to_dict()}
        


def load_experiment_dfs(config):
    experiment_dfs = {}
    df  = get_db_table(config.table_name,config.db_path)
    #df = df[pandaconfig.lower_limit_df:pandaconfig.upper_limit_df]
    for name, df in df.groupby('split'):
        experiment_dfs[name] = df.set_index("image_id")
    return experiment_dfs

def load_coords_df(experiment_dfs,splits=["train", "val", "test"]):
    patchmaskpreparer = PatchMaskPreparer(pandaconfig)
    coords_df = pd.DataFrame()
    for split in splits:
        tmp_filenames = [f"{index}.tiff" for index in experiment_dfs[split].index]
        len(tmp_filenames)
        tmp_df = patchmaskpreparer.get_patch_coords(tmp_filenames).assign(split=split)
        coords_df = pd.concat([coords_df, tmp_df])
    return coords_df

def losses(coords_df):
    df = coords_df
    #len_training = len(df[df['split'] == 'train'].value_counts())
    neg_val = df[df['split'] == 'train']['label'].value_counts()[0]
    pos_val = df[df['split'] == 'train']['label'].value_counts()[1]
    weight_loss = torch.tensor(neg_val/pos_val)

    return weight_loss


if __name__ == '__main__':
    experiment_dfs = load_experiment_dfs(pandaconfig)
    coords_df = load_coords_df(experiment_dfs)
    print('coords',coords_df)

    #image_transforms_train = image_transforms(augment=True)
    #image_transforms_val = image_transforms(augment=False)
    data_module = TrainDataPanda(pandaconfig,coords_df)
    train_loader = DataLoader(data_module,batch_size=256,shuffle=False)

    for idx,(patch,label,dict) in enumerate(train_loader):
        print(dict)


    
    