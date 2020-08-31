import random

import numpy as np
import torch
import torchvision.transforms as T
import PIL
from PIL import ImageDraw
import config
from scipy.spatial.transform import Rotation as R


class RandomSwitchImages:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, item):
        if self.p < random.random():
            return item
        (img1, img2), (t_gt, r_gt), (pts1, pts2), (K1, K2) = item
        new_r = r_gt.inv()
        new_t = new_r.apply(-t_gt)
        return (img2, img1), (new_t, new_r), (pts2, pts1), (K2, K1)

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


class ToTensor:

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, item):
        (img1, img2), (t_gt, r_gt), (pts1, pts2), (K1, K2) = item
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        t_gt = torch.from_numpy(t_gt.astype(np.float32))
        r_gt = torch.from_numpy(r_gt.as_quat().astype(np.float32))
        pts1 = torch.from_numpy(pts1).float()
        pts2 = torch.from_numpy(pts2).float()
        K1 = torch.from_numpy(K1).float()
        K2 = torch.from_numpy(K2).float()

        return (img1, img2), (t_gt, r_gt), (pts1, pts2), (K1, K2)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImageTransform:

    def __init__(self, t):
        self.t = t

    def __call__(self, item):
        (img1, img2), gt, pts, ks = item
        img1 = self.t(img1)
        img2 = self.t(img2)
        return (img1, img2), gt, pts, ks

    def __repr__(self):
        return f'{self.__class__.__name__}({self.t})'


class ImageNormalize(ImageTransform):

    def __init__(self, stats=None):
        if stats is None:
            stats = config.imagenet_stats
        super().__init__(T.Normalize(*stats))


to_tensor_and_normalize = T.Compose([ToTensor(), ImageNormalize()])
