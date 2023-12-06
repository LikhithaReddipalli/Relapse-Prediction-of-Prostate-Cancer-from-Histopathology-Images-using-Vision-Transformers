import wandb
import os
import configs.config as config
import time
wandb.login(key = '585bb9e7f76b06ff3cd69aaeaf08f3c04f438985')
from src.models.vit_pretrain import PretrainPanda
from src.models.vit_pretrain_eval import PandaEval
from src.models.vit_mil_patch import VitSurvPatch
from src.models.vit_hier_2k import VitSurvHier2k
from src.models.vit_hier_2k_eval import VitSurvHier2kEval
from src.models.cnn_mil_patch import CNNSurvPatch
from src.models.vit_mil_patch_eval import VitSurvPatchEval




# main run file

file_name = os.path.join("/data/PANDA/code/survival_prediction/Experiments" , time.strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir(file_name)
wandb.init(project = config.model, dir=file_name, name = config.model + " : " + time.strftime("%Y_%m_%d_%H_%M_%S") )
config_var = {}
for name in dir(config):
    if not name.startswith("__"):
        config_var[name] = getattr(config,name)

wandb.config.update(config_var)
train = PretrainPanda(file_name ,config)


