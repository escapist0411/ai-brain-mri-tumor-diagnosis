import numpy as np
import torch
from torch.utils.data import Dataset


class MRISegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.images = np.load(f"{data_dir}/images.npy")

        # Dummy masks (prototype)
        self.masks = self.create_dummy_masks()

    def create_dummy_masks(self):
        masks = []
        for img in self.z images:
            mask = np.zeros((224, 224), dtype=np.float32)

            rr, cc = np.ogrid[:224, :224]
            center = (112, 112)
            radius = 40

            circle = (rr - center[0])**2 + (cc - center[1])**2 < radius**2
            mask[circle] = 1.0
            masks.append(mask)

        return np.array(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert RGB → grayscale
        image = np.mean(image, axis=2)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32).unsqueeze(0)

        return image, mask
