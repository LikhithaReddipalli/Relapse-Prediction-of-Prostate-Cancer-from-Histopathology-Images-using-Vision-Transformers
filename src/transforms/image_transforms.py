import random
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
#from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from torchvision.transforms import (
    AugMix,
    AutoAugment,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    Normalize,
    transforms,
)


class RandomDiscreteRotation(nn.Module):
    def __init__(self, angles: Sequence[int] = [0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
def numpy_to_tensor(x):
    return torch.from_numpy(x)  #lambda x: torch.from_numpy(x)


# TODO: This could be Lambda
def div_by_255(x):
    return x / 255   #lambda x:x / 255 


class ImageTransforms(transforms.Compose):
    def __init__(self, config, augment: bool = False, normalize: bool = False ):

        transforms = [numpy_to_tensor]

        if config.resize_size is not None:
            #print('---test--resize--')
            transforms.extend(
                [
                    Resize(size=config.resize_size),
                ]
            )

        if augment:
            transforms.extend(
                [
                    RandomDiscreteRotation([0, 90, 180, 270]),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                ]
            )

        transforms.extend(
            [
                div_by_255,
            ]
        )

        if normalize:

            transforms.extend(
                [
                    Normalize(mean=config.mean, std=config.std),
                ]
            )

        super().__init__(transforms)