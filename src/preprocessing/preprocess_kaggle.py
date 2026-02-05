import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_SIZE = 224
CLASSES = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load image, resize, normalize.
    Returns: (224, 224, 3) float32 array
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_split(split: str):
    """
    split: 'training' or 'testing'
    """
    input_dir = Path("data/raw/kaggle") / split
    output_dir = Path("data/processed/kaggle") / split

    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    for cls_name, cls_idx in CLASSES.items():
        cls_dir = input_dir / cls_name

        for img_file in tqdm(list(cls_dir.iterdir()), desc=f"{split}/{cls_name}"):
            img = preprocess_image(str(img_file))
            images.append(img)
            labels.append(cls_idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    np.save(output_dir / "images.npy", images)
    np.save(output_dir / "labels.npy", labels)

    print(f"Saved {split} data:")
    print(" Images:", images.shape)
    print(" Labels:", labels.shape)


if __name__ == "__main__":
    preprocess_split("training")
    preprocess_split("testing")
