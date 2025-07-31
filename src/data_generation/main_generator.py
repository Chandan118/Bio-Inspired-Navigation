import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import Config
from src.utils import ensure_dir
from .oct_simulator import simulate_oct
from .lsci_simulator import simulate_lsci
from .noise_models import add_gaussian_noise, add_motion_blur


random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


def _random_mask(img_size):
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.uint8)
    # Draw random rectangles as defects
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
        x2, y2 = random.randint(x1 + 20, w), random.randint(y1 + 20, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    return mask


def _save_pair(img: np.ndarray, mask: np.ndarray, img_path: Path, mask_path: Path):
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask * 255)


def generate_dataset():
    # Ensure dirs
    for p in [Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, Config.VAL_IMG_DIR, Config.VAL_MASK_DIR]:
        ensure_dir(p)

    settings = [
        (Config.TRAIN_SAMPLES, Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR),
        (Config.VAL_SAMPLES, Config.VAL_IMG_DIR, Config.VAL_MASK_DIR),
    ]

    for n_samples, img_dir, mask_dir in settings:
        for idx in tqdm(range(n_samples), desc=f"Generating data -> {img_dir.parent.name}"):
            # generate random mask
            mask = _random_mask(Config.OCT_IMG_SIZE)
            modality = random.choice(["oct", "lsci"])
            if modality == "oct":
                img = simulate_oct(mask, Config.OCT_IMG_SIZE)
            else:
                img = simulate_lsci(mask, Config.LSCI_IMG_SIZE)

            img = add_gaussian_noise(img, Config.GAUSSIAN_NOISE_STD)
            img = add_motion_blur(img, Config.MOTION_BLUR_KERNEL)

            _save_pair(img, mask, img_dir / f"{idx:04d}.png", mask_dir / f"{idx:04d}.png")

    print("Synthetic dataset ready →", Config.DATA_DIR)