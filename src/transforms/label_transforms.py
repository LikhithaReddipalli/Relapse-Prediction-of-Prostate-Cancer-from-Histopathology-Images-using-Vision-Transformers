import torch
from torchvision.transforms import transforms


def to_long_tensor(x):
    return torch.tensor(x, dtype=torch.long)


def to_float_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


class LabelTransformsClas(transforms.Compose):
    def __init__(self):
        transforms = [
            to_long_tensor,
            torch.squeeze,
        ]
        super().__init__(transforms)


class LabelTransformsSurv(transforms.Compose):
    def __init__(self):
        transforms = [
            to_float_tensor,
            torch.squeeze,
        ]
        super().__init__(transforms)
