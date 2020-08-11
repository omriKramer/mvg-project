import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

import config


class ImagePair:
    # Initialize Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        config.imgs_normalization,
    ])

    def __init__(self, pair_path):
        # Get GT
        gt_path = os.path.join(pair_path, 'GT')
        self.R_GT = R.from_matrix(np.load(os.path.join(gt_path, 'GT_R12.npy')))
        self.t_GT = np.load(os.path.join(gt_path, 'GT_t12.npy')).flatten()

        # Get inputs
        inputs_path = os.path.join(pair_path, 'inputs')
        self.K1 = np.load(os.path.join(inputs_path, 'K1.npy'))
        self.K2 = np.load(os.path.join(inputs_path, 'K2.npy'))

        # TODO: Normalize points if using
        self.pts1 = np.load(os.path.join(inputs_path, 'points1.npy'))
        self.pts2 = np.load(os.path.join(inputs_path, 'points2.npy'))

        self.im1 = cv2.imread(os.path.join(inputs_path, 'im1.jpg'))
        self.im2 = cv2.imread(os.path.join(inputs_path, 'im2.jpg'))

    @property
    def data(self):
        x = self.transform(self.im1), self.transform(self.im2)
        y = torch.from_numpy(self.t_GT).float(), torch.from_numpy(self.R_GT.as_quat()).float(),
        return x, y


class ImagePairsDataset(Dataset):
    def __init__(self, items, path):
        # Initialize variable
        self.path = Path(path)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pair_path = self.path / self.items[idx]
        image_pair = ImagePair(pair_path)
        return image_pair.data


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


def trainval_ds(path, csv_path, **kwargs):
    df = pd.read_csv(csv_path)
    is_val = df['is_val']
    train_items = df['item'][~is_val].tolist()
    val_items = df['item'][is_val].tolist()

    train_ds = ImagePairsDataset(train_items, path)
    val_ds = ImagePairsDataset(val_items, path)
    data = Data(train_ds, val_ds, **kwargs)
    return data
