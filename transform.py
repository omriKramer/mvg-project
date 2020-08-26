import random

import numpy as np
import torch
import torchvision.transforms as T

import config


class RandomSwitchImages:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, item):
        if self.p < random.random():
            return item
        (img1, img2), (t_gt, r_gt) = item
        new_r = r_gt.inv()
        new_t = new_r.apply(-t_gt)
        return (img2, img1), (new_t, new_r)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class ToTensor:

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, item):
        (img1, img2), (t_gt, r_gt) = item
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        t_gt = torch.from_numpy(t_gt.astype(np.float32))
        r_gt = torch.from_numpy(r_gt.as_quat().astype(np.float32))
        return (img1, img2), (t_gt, r_gt)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImageTransform:

    def __init__(self, t):
        self.t = t

    def __call__(self, item):
        (img1, img2), gt = item
        img1 = self.t(img1)
        img2 = self.t(img2)
        return (img1, img2), gt

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t})'


class ImageNormalize(ImageTransform):

    def __init__(self, stats=None):
        if stats is None:
            stats = config.imagenet_stats
        super().__init__(T.Normalize(*stats))


to_tensor_and_normalize = T.Compose([ToTensor(), ImageNormalize()])
