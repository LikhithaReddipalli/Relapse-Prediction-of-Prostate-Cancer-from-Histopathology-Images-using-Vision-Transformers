import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import random, os
import numpy as np
import torch

def show_full_image(image):
    image = image.permute(1,2,0)
    plt.imshow(image)
    plt.show()


def show_patch_image(image, img_size, patch_size):
    '''
    Display patches in the image of vision transformer
    '''
    image = image.permute(1,2,0)
    plt.figure(figsize=(patch_size/16, patch_size/16))
    plt.imshow(image[:patch_size, :, :])
    plt.show()
    num_patches = image.shape[0]/patch_size 
    assert image.shape[0] % patch_size == 0, "Image size must be divisible by patch size" 
    print(f"Number of patches per row: {num_patches}\nPatch size: {patch_size} pixels x {patch_size} pixels")

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=1, 
                            ncols=image.shape[0]// patch_size, # one column for each patch
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)

    # Iterate through number of patches in the top row
    for i, patch in enumerate(range(0, image.shape[0], patch_size)):
        axs[i].imshow(image[208:223,:][:patch_size, patch:patch+patch_size, :]); # keep height index constant, alter the width index
        axs[i].set_xlabel(i+183) # set the label
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()


def ten_to_lst(lst):
    new_gt = []
    for i in range(0, len(lst)):
        new_gt.append(lst[i].tolist())
    #print(flatten(new_gt))

    new_gt = flatten(new_gt)

    return new_gt


def flatten(lst):
    """
    Flatten a nested list.
    """

    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def append_suffix_before_ext(filename, suffix):
    return os.path.splitext(filename)[0] + suffix + os.path.splitext(filename)[1]


def seed_everything(seed: int):
    '''
    set seeds
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
