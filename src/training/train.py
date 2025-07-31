from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.model.dataset import SegmentationDataset
from src.model.unet_model import UNet
from src.model.loss import BCEDiceLoss
from src.utils import ensure_dir


def train():
    ensure_dir(Config.MODELS_DIR)
    train_ds = SegmentationDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR)
    val_ds = SegmentationDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

    model = UNet(in_channels=1, out_channels=1).to(Config.DEVICE)
    loss_fn = BCEDiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)

    best_dice = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
            imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Train loss: {epoch_loss / len(train_loader):.4f}")

        # Validation Dice
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
                preds = model(imgs)
                preds_bin = (preds > 0.5).float()
                intersection = (preds_bin * masks).sum(dim=(1, 2, 3))
                dice = (2 * intersection) / (preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
                dice_scores.extend(dice.cpu().numpy())
        mean_dice = sum(dice_scores) / len(dice_scores)
        print(f"Val Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), Config.MODELS_DIR / "best_defect_detector.pth")
            print("Saved new best model →", best_dice)

    print("Training complete. Best Dice:", best_dice)