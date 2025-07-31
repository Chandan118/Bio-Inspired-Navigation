from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None):
        self.images = sorted(list(images_dir.glob("*.png")))
        self.masks = sorted(list(masks_dir.glob("*.png")))
        self.transform = transform
        assert len(self.images) == len(self.masks), "Images and masks mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32") / 255.0
        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Add channel dim
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return image, mask