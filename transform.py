import random

import numpy as np
import torch
import torchvision.transforms as T

import config


def inverse_relative_pose(t, rot):
    new_rot = rot.inv()
    new_t = new_rot.apply(-t)
    return new_t, new_rot


class RandomSwitchImages:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, item):
        if self.p < random.random():
            return item
        (img1, img2), (t_gt, r_gt), *other = item
        new_pos = inverse_relative_pose(t_gt, r_gt)
        result = (img2, img1), new_pos
        return result + tuple(other)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


def to_tensor_iterable(x):
    if not isinstance(x, (list, tuple)):
        return torch.from_numpy(x.astype(np.float32))

    return tuple(to_tensor_iterable(e) for e in x)


class ToTensor:

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, item):
        (img1, img2), (t_gt, r_gt), *other = item
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        t_gt = torch.from_numpy(t_gt.astype(np.float32))
        r_gt = torch.from_numpy(r_gt.as_quat().astype(np.float32))
        other = to_tensor_iterable(other)
        return ((img1, img2), (t_gt, r_gt)) + other

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImageTransform:

    def __init__(self, t):
        self.t = t

    def __call__(self, item):
        (img1, img2), *other = item
        img1 = self.t(img1)
        img2 = self.t(img2)
        return ((img1, img2),) + tuple(other)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t})'


class ImageNormalize(ImageTransform):

    def __init__(self, stats=None):
        if stats is None:
            stats = config.imagenet_stats
        super().__init__(T.Normalize(*stats))


to_tensor_and_normalize = T.Compose([ToTensor(), ImageNormalize()])


class ColorJitter(ImageTransform):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(T.ColorJitter(brightness, contrast, saturation, hue))
