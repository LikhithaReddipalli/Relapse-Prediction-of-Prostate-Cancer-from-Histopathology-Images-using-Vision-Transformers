model = "Finale_hier"

#base_path_surv = '/home/likhitha/pca/survival'  # --lab 
#base_path_surv = 'C:/Users/likhi/Desktop/Thesis/pc/survival/'
#data_dir_surv = '/home/likhitha/pca/survival' # --lab

data_dir_surv = '/data/PANDA/survival/png_fitted_center'

db_path = "survival_prediction\db_lr.sqlite"
#db_path = "survival_prediction\db.sqlite"

data_augmentation =  True
degrees = [90,180,270,360]

num_classes = 1
num_channels = 3

rescale_factor = [0.0, 1.0]
#rescale_factor = [0.109375,0.109375,1]
img_size = 1024 #2048
resize = 1024 #224
patch_size = 256 #512
num_patches = 16

embedding_dim = 768
mlp_size = 3072 
num_transformer_layers = 12 
num_heads = 12 
dropout = 0.1 

lr  = 3e-6
betas = (0.9, 0.999)
weight_decay = 0.03
mean = (0.5,0.5,0.5) #(0.485, 0.456, 0.406)    # (0.5,0.5,0.5) - chen
std =  (0.5, 0.5, 0.5) #(0.229, 0.224, 0.225)    # (0.5, 0.5, 0.5) - chen
embedding_ip_4k = 192
num_workers = 18



device = "cuda"
batch_size = 1
epochs = 200
seed = 5

validation = True
pretrain = False
test = True
onlyEval = False

checkpoint =  '/data/PANDA/code/survival_prediction/Experiments/2023_07_19_18_09_03/checkpoints_300.pth' #pretrain weights model PANDA


#checkpoint =  '/data/PANDA/code/survival_prediction/Experiments/2023_09_02_00_58_58/checkpoints_14.pth'
#'survival_prediction\\Experiments\\2023_03_22_12_09_56\\checkpoints%5C25.pth' #-without pretrain weights model
