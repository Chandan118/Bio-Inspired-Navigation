from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.model.dataset import SegmentationDataset
from src.model.unet_model import UNet
from src.utils import ensure_dir, dice_coef, iou_score, save_overlay


def evaluate():
    ensure_dir(Config.OUTPUT_DIR)

    dataset = SegmentationDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR)
    loader = DataLoader(dataset, batch_size=1)

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(Config.MODELS_DIR / "best_defect_detector.pth", map_location="cpu"))
    model.eval()

    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            preds = model(imgs)
            preds_bin = (preds > 0.5).float()
            pred_np = preds_bin.squeeze().cpu().numpy()
            mask_np = masks.squeeze().cpu().numpy()
            dices.append(dice_coef(mask_np, pred_np))
            ious.append(iou_score(mask_np, pred_np))

    report_path = Config.REPORT_PATH
    with open(report_path, "w") as f:
        f.write("AutoOpticalDiagnostics – Evaluation Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mean Dice: {np.mean(dices):.4f}\n")
        f.write(f"Mean IoU:  {np.mean(ious):.4f}\n")
    print("Saved report →", report_path)

    # save one sample visual
    imgs, masks = next(iter(loader))
    preds = model(imgs)
    preds_bin = (preds > 0.5).float()
    img_np = (imgs.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_np = masks.squeeze().cpu().numpy().astype(np.uint8)
    pred_np = preds_bin.squeeze().cpu().numpy().astype(np.uint8)

    save_overlay(img_np, mask_np, pred_np, Config.SAMPLE_PREDICTION_PATH)
    print("Saved overlay →", Config.SAMPLE_PREDICTION_PATH)