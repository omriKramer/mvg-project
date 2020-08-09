import torch
import config
import os
import numpy as np
import cv2
import torchvision.transforms as transforms


class ImagePair():
    def __init__(self, pair_path):
        # Initialize Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            config.imgs_normalization,
        ])

        # Get GT
        GT_path = os.path.join(pair_path, 'GT')
        self.R_GT = torch.from_numpy(np.load(os.path.join(GT_path, 'GT_R12.npy'))).float()
        self.t_GT = torch.from_numpy(np.load(os.path.join(GT_path, 'GT_t12.npy'))).float()

        # Get inputs
        inputs_path = os.path.join(pair_path, 'inputs')
        self.K1 = torch.from_numpy(np.load(os.path.join(inputs_path, 'K1.npy'))).float()
        self.K2 = torch.from_numpy(np.load(os.path.join(inputs_path, 'K2.npy'))).float()

        # TODO Normalize points if using
        self.pts1 = torch.from_numpy(np.load(os.path.join(inputs_path, 'points1.npy'))).float()
        self.pts2 = torch.from_numpy(np.load(os.path.join(inputs_path, 'points2.npy'))).float()

        self.im1 = self.transform(cv2.imread(os.path.join(inputs_path, 'im1.jpg')))
        self.im2 = self.transform(cv2.imread(os.path.join(inputs_path, 'im2.jpg')))


class ImagePairsScene():
    def __init__(self, path):
        self.path = path
        self.len = len(os.listdir(path))

    def get_len(self):
        return self.len

    def get_pair(self, idx):
        return ImagePair(os.path.join(self.path, str(idx + 1)))


class ImagePairsDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Initialize variable
        self.dataset_path = config.dataset_path
        scenes_list = os.listdir(self.dataset_path)
        self.scene_indices = np.zeros(0, dtype=np.int)
        self.scenes = []
        curr_idx = 0

        for scene_path in scenes_list:
            scene = ImagePairsScene(os.path.join(self.dataset_path, scene_path))
            self.scene_indices = np.append(self.scene_indices, curr_idx + scene.get_len())
            self.scenes.append(scene)
            curr_idx += scene.get_len()

        self.len = curr_idx

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        scene_idx = np.argwhere(self.scene_indices > idx)[0][0]
        if scene_idx != 0:
            pair_idx = idx - self.scene_indices[scene_idx - 1]
        else:
            pair_idx = idx

        return self.scenes[scene_idx].get_pair(pair_idx)


