import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


def ensure_dir(path: Path):
    """Create directory (and parents) if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------- METRICS ---------------------- #

def dice_coef(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + eps) / (y_true_f.sum() + y_pred_f.sum() + eps)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + eps) / (union + eps)


# ---------------------- VISUALS ---------------------- #

def save_overlay(image: np.ndarray, mask: np.ndarray, pred: np.ndarray, dest: Path):
    """Create a side-by-side visualisation of GT vs prediction."""
    h, w = image.shape[:2]
    colour_mask_gt = np.zeros((h, w, 3))
    colour_mask_gt[:, :, 1] = mask * 255  # green GT
    colour_mask_pred = np.zeros((h, w, 3))
    colour_mask_pred[:, :, 2] = pred * 255  # red pred

    over_gt = cv2.addWeighted(image, 0.7, colour_mask_gt.astype(np.uint8), 0.3, 0)
    over_pred = cv2.addWeighted(image, 0.7, colour_mask_pred.astype(np.uint8), 0.3, 0)
    concat = np.concatenate([over_gt, over_pred], axis=1)
    cv2.imwrite(str(dest), concat)