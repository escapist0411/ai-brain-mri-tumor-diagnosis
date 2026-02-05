import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class GANDataset(Dataset):
    def __init__(self, folder_path, img_size=64):
        self.folder_path = folder_path
        self.img_size = img_size
        self.images = os.listdir(folder_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.images[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize between -1 and 1 (GAN standard)
        img = (img.astype(np.float32) / 127.5) - 1.0

        img = torch.tensor(img).unsqueeze(0)

        return img
