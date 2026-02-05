import numpy as np
import torch
from torch.utils.data import Dataset


class KaggleBrainMRIDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        data_dir example:
        data/processed/kaggle/training
        """
        self.images = np.load(f"{data_dir}/images.npy")
        self.labels = np.load(f"{data_dir}/labels.npy")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]      # (224, 224, 3)
        label = self.labels[idx]

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)

        # Change shape: (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
