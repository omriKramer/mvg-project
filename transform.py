import random

import numpy as np
import torch
import torchvision.transforms as T
import PIL
from PIL import ImageDraw
import config
from scipy.spatial.transform import Rotation as R


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


class RandomRotation:
    def __init__(self, p=0.5, max_angle=45):
        self.p = p
        self.max_angle = max_angle

    def __call__(self, item):
        if self.p < random.random():
            return item

        (img1, img2), (t_gt, r_gt), (pts1, pts2), (K1, K2) = item
        angle = random.uniform(-self.max_angle, self.max_angle)

        # ~~~~~~~~~~~~~~~~~~ Sanity Check ~~~~~~~~~~~~~~~~~~~~~~~~
        # angle=45
        # draw = ImageDraw.Draw(img2)
        # draw.ellipse(np.concatenate((pts2[0:2, 0].T, pts2[0:2, 0].T + 5)).flatten().tolist(), fill='red', outline='red')
        # draw.ellipse(np.concatenate((pts2[0:2, 1].T, pts2[0:2, 1].T + 5)).flatten().tolist(), fill='red', outline='red')
        # draw.ellipse(np.concatenate((pts2[0:2, 2].T, pts2[0:2, 2].T + 5)).flatten().tolist(), fill='red', outline='red')
        # img2.save('tmp.jpg', 'JPEG')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Rotate Image
        img2 = T.functional.rotate(img2, angle, resample=PIL.Image.BICUBIC, center=(K2[0,2], K2[1,2]))

        # Rotate r_gt and points
        angle_rot = R.from_euler('z', -angle, degrees=True)
        r_gt = r_gt * angle_rot.inv()
        pts2 = K2 @ angle_rot.as_matrix() @ np.linalg.inv(K2) @ pts2

        # ~~~~~~~~~~~~~~~~~~ Sanity Check ~~~~~~~~~~~~~~~~~~~~~~~~
        # draw = ImageDraw.Draw(img2)
        # draw.ellipse(np.concatenate((pts2[0:2, 0].T, pts2[0:2, 0].T + 5)).flatten().tolist(), fill='blue', outline='blue')
        # draw.ellipse(np.concatenate((pts2[0:2, 1].T, pts2[0:2, 1].T + 5)).flatten().tolist(), fill='blue', outline='blue')
        # draw.ellipse(np.concatenate((pts2[0:2, 2].T, pts2[0:2, 2].T + 5)).flatten().tolist(), fill='blue', outline='blue')
        # img2.save('tmp_rotated.jpg', 'JPEG')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return (img1, img2), (t_gt, r_gt), (pts1, pts2), (K1, K2)

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
