#db_path = "survival_prediction/db.sqlite"
db_path = "/data/PANDA/code/survival_prediction/db.sqlite"
table_name = 'panda.experiments.tissue'
lower_limit_df = 0
upper_limit_df = 7

#image_base_dir = 'survival_prediction/data/panda/train_images'
#mask_base_dir = 'survival_prediction/data/panda/train_label_masks_new_4chan'

image_base_dir = '/data/PANDA/PANDA/train_images'
mask_base_dir = '/data/PANDA/PANDA/train_label_masks_new_4chan'


patch_size= 256
fg_mask_channel=3
fg_mask_invert=False
fg_mask_threshold=0.1
mask_downsample_rate=16
label_mask_channel=2
label_mask_invert=False
label_mask_threshold=0.9
mask_filename_suffix="_mask"
seed = 42
undersample_majority_label= False
stage = 'train'
mean  = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
resize_size= None
normalize = True