import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

import transform


def get_data(pair_path):
    # Get GT
    gt_path = os.path.join(pair_path, 'GT')
    r_gt = R.from_matrix(np.load(os.path.join(gt_path, 'GT_R12.npy')))
    t_gt = np.load(os.path.join(gt_path, 'GT_t12.npy')).flatten()

    # Get inputs
    inputs_path = os.path.join(pair_path, 'inputs')
    im1 = Image.open(os.path.join(inputs_path, 'im1.jpg'))
    im2 = Image.open(os.path.join(inputs_path, 'im2.jpg'))

    return (im1, im2), (t_gt, r_gt)


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
