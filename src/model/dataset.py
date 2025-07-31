"""
Dataset classes for optical imaging data with comprehensive data augmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import logging


class OpticalDataset(Dataset):
    """
    Dataset class for optical imaging data with comprehensive augmentation.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        target_transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        mode: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            transform: Image transformations
            target_transform: Mask transformations
            image_size: Target image size
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.mode = mode
        
        # Get image and mask file paths
        self.images = sorted(list(self.images_dir.glob("*.png")))
        self.masks = sorted(list(self.masks_dir.glob("*_mask.png")))
        
        # Validate dataset
        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.images)} samples from {mode} dataset")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and mask tensors
        """
        # Load image and mask
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.uint8)  # Binarize mask
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if self.target_transform is not None:
            mask = self.target_transform(image=mask)['image']
        
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path)
        }


class OpticalDatasetWithMetadata(Dataset):
    """
    Enhanced dataset class with metadata support.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        transform: Optional[A.Compose] = None,
        target_transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        mode: str = "train"
    ):
        """
        Initialize dataset with metadata.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            metadata_file: Path to metadata JSON file
            transform: Image transformations
            target_transform: Mask transformations
            image_size: Target image size
            mode: Dataset mode
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.mode = mode
        
        # Load metadata if available
        self.metadata = []
        if self.metadata_file and self.metadata_file.exists():
            import json
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Get file paths
        self.images = sorted(list(self.images_dir.glob("*.png")))
        self.masks = sorted(list(self.masks_dir.glob("*_mask.png")))
        
        # Validate dataset
        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) != number of masks ({len(self.masks)})")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.images)} samples from {mode} dataset")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with metadata.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, mask, and metadata
        """
        # Load image and mask
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.uint8)  # Binarize mask
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if self.target_transform is not None:
            mask = self.target_transform(image=mask)['image']
        
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Get metadata if available
        sample_metadata = {}
        if idx < len(self.metadata):
            sample_metadata = self.metadata[idx]
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'metadata': sample_metadata
        }


def get_training_transforms(
    image_size: Tuple[int, int] = (512, 512),
    augmentation_probability: float = 0.8
) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: Target image size
        augmentation_probability: Probability of applying augmentations
        
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.5),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.5),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.5),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], p=augmentation_probability)


def get_validation_transforms(
    image_size: Tuple[int, int] = (512, 512)
) -> A.Compose:
    """
    Get validation data transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (512, 512)
) -> A.Compose:
    """
    Get test data transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_data_loaders(
    train_images_dir: Union[str, Path],
    train_masks_dir: Union[str, Path],
    val_images_dir: Union[str, Path],
    val_masks_dir: Union[str, Path],
    train_metadata_file: Optional[Union[str, Path]] = None,
    val_metadata_file: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    augmentation_probability: float = 0.8,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_images_dir: Training images directory
        train_masks_dir: Training masks directory
        val_images_dir: Validation images directory
        val_masks_dir: Validation masks directory
        train_metadata_file: Training metadata file
        val_metadata_file: Validation metadata file
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        augmentation_probability: Augmentation probability
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transforms = get_training_transforms(image_size, augmentation_probability)
    val_transforms = get_validation_transforms(image_size)
    
    # Create datasets
    train_dataset = OpticalDatasetWithMetadata(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        metadata_file=train_metadata_file,
        transform=train_transforms,
        image_size=image_size,
        mode="train"
    )
    
    val_dataset = OpticalDatasetWithMetadata(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        metadata_file=val_metadata_file,
        transform=val_transforms,
        image_size=image_size,
        mode="val"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


class BalancedSampler(torch.utils.data.Sampler):
    """
    Balanced sampler for handling class imbalance.
    """
    
    def __init__(self, dataset: Dataset, replacement: bool = True):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights based on mask statistics."""
        weights = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            mask = sample['mask']
            
            # Calculate positive pixel ratio
            positive_ratio = mask.sum() / mask.numel()
            weight = 1.0 / (positive_ratio + 1e-8)
            weights.append(weight)
        
        return torch.tensor(weights)
    
    def __iter__(self):
        """Return iterator of indices."""
        return iter(torch.multinomial(self.class_weights, len(self.dataset), self.replacement).tolist())
    
    def __len__(self):
        return len(self.dataset)


def create_balanced_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create data loader with balanced sampling.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader with balanced sampling
    """
    sampler = BalancedSampler(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )