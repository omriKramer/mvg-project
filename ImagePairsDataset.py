import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
import transform
import config
import utils


def get_data(pair_path):
    # Get GT
    gt_path = os.path.join(pair_path, 'GT')
    r_gt = R.from_matrix(np.load(os.path.join(gt_path, 'GT_R12.npy')))
    t_gt = np.load(os.path.join(gt_path, 'GT_t12.npy')).flatten()

    # Get inputs
    inputs_path = os.path.join(pair_path, 'inputs')
    im1 = Image.open(os.path.join(inputs_path, 'im1.jpg'))
    im2 = Image.open(os.path.join(inputs_path, 'im2.jpg'))

    # Get inputs
    inputs_path = os.path.join(pair_path, 'inputs')
    K1 = np.load(os.path.join(inputs_path, 'K1.npy'))
    K2 = np.load(os.path.join(inputs_path, 'K2.npy'))

    # TODO: Normalize points if using in network
    pts1, pts2 = get_points(inputs_path, r_gt, t_gt, K1, K2)

    return (im1, im2), (t_gt, r_gt), (pts1, pts2), (K1, K2)


def get_points(path, R_gt, t_gt, K1_gt, K2_gt):
    pts1 = np.load(os.path.join(path, 'points1.npy'))
    pts2 = np.load(os.path.join(path, 'points2.npy'))

    F = utils.get_fundamental_mat(torch.from_numpy(t_gt).float().unsqueeze(0),
                                   torch.from_numpy(R_gt.as_quat()).float().unsqueeze(0),
                                   torch.from_numpy(K1_gt).float().unsqueeze(0),
                                   torch.from_numpy(K2_gt).float().unsqueeze(0)).numpy().squeeze()
    pts1, pts2 = cv2.correctMatches(F, pts1[0:2, :, np.newaxis].T, pts2[0:2, :, np.newaxis].T)
    # pts1 = np.vstack((pts1.squeeze().T, np.ones((1, pts1.shape[1]))))
    # pts2 = np.vstack((pts2.squeeze().T, np.ones((1, pts2.shape[1]))))
    #
    # pts2 = np.concatenate((pts2, np.zeros((3, config.max_num_of_points - pts2.shape[1]))), axis=1)
    # pts1 = np.concatenate((pts1, np.zeros((3, config.max_num_of_points - pts1.shape[1]))), axis=1)

    pts1 = pts1.squeeze().T
    pts2 = pts2.squeeze().T

    if config.max_num_of_points - pts2.shape[1] < 0:
        print(pts2.shape[1])

    pts2 = np.concatenate((pts2, np.zeros((2, config.max_num_of_points - pts2.shape[1]))), axis=1)
    pts1 = np.concatenate((pts1, np.zeros((2, config.max_num_of_points - pts1.shape[1]))), axis=1)

    pts1 = np.vstack((pts1, np.ones((1, pts1.shape[1]))))
    pts2 = np.vstack((pts2, np.ones((1, pts2.shape[1]))))

    return pts1, pts2


class ImagePairsDataset(Dataset):
    def __init__(self, items, path, tfm=transform.to_tensor_and_normalize):
        self.path = Path(path)
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair_path = self.path / self.items[idx]
        item = get_data(pair_path)
        if self.tfm:
            item = self.tfm(item)
        return item


class Data:

    def __init__(self, train_ds, valid_ds, bs=16, num_workers=16):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.bs = bs
        self.num_workers = min(num_workers, bs)

    def _dl(self, ds, **kwargs):
        bs = kwargs.pop('batch_size', self.bs)
        num_workers = kwargs.pop('num_workers', self.num_workers)
        return DataLoader(ds, batch_size=bs, num_workers=num_workers, **kwargs)

    def train_dl(self, shuffle=True, **kwargs):
        return self._dl(self.train_ds, shuffle=shuffle, **kwargs)

    def valid_dl(self, shuffle=False, **kwargs):
        return self._dl(self.valid_ds, shuffle=shuffle, **kwargs)


def trainval_ds(path, csv_path, train_tfm=transform.to_tensor_and_normalize, **kwargs):
    df = pd.read_csv(csv_path)
    is_val = df['is_val']
    train_items = df['item'][~is_val].tolist()
    val_items = df['item'][is_val].tolist()

    train_ds = ImagePairsDataset(train_items, path, tfm=train_tfm)
    val_ds = ImagePairsDataset(val_items, path)
    data = Data(train_ds, val_ds, **kwargs)
    return data
