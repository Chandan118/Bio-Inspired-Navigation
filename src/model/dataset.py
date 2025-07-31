"""
Dataset classes for optical diagnostics data.
Implements advanced data loading, preprocessing, and augmentation techniques.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

class OpticalDiagnosticsDataset(Dataset):
    """
    Dataset class for optical diagnostics data.
    
    Features:
    - Multi-modal data loading (LSCI, OCT, hybrid)
    - Advanced data augmentation
    - Flexible data organization
    - Metadata tracking
    - Quality control
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_transform: Optional[A.Compose] = None,
        modalities: List[str] = None,
        environments: List[str] = None,
        materials: List[str] = None,
        defect_types: List[str] = None,
        max_samples: Optional[int] = None,
        balance_classes: bool = True,
        cache_data: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.modalities = modalities or ["lsci", "oct"]
        self.environments = environments
        self.materials = materials
        self.defect_types = defect_types
        self.max_samples = max_samples
        self.balance_classes = balance_classes
        self.cache_data = cache_data
        
        # Data cache
        self.cached_images = {}
        self.cached_masks = {}
        self.cached_metadata = {}
        
        # Load data samples
        self.samples = self._load_samples()
        
        # Apply class balancing if requested
        if balance_classes and split == "train":
            self.samples = self._balance_classes()
        
        # Apply max samples limit
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Modalities: {self.modalities}")
        logger.info(f"Environments: {self.environments}")
        logger.info(f"Materials: {self.materials}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load data samples based on configuration."""
        samples = []
        
        # Load metadata files
        metadata_dir = self.data_dir / "data" / "synthetic" / self.split / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"Metadata directory not found: {metadata_dir}")
            return samples
        
        metadata_files = list(metadata_dir.glob("*.json"))
        
        for metadata_file in tqdm(metadata_files, desc=f"Loading {self.split} samples"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Apply filters
                if not self._passes_filters(metadata):
                    continue
                
                # Get file paths
                image_path = metadata.get('image_path')
                mask_path = metadata.get('mask_path')
                
                if not image_path or not mask_path:
                    continue
                
                # Verify files exist
                if not Path(image_path).exists() or not Path(mask_path).exists():
                    continue
                
                # Add to samples
                sample = {
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'metadata': metadata
                }
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")
                continue
        
        return samples
    
    def _passes_filters(self, metadata: Dict[str, Any]) -> bool:
        """Check if sample passes all filters."""
        # Modality filter
        if self.modalities and metadata.get('modality') not in self.modalities:
            return False
        
        # Environment filter
        if self.environments and metadata.get('environment') not in self.environments:
            return False
        
        # Material filter
        if self.materials and metadata.get('material') not in self.materials:
            return False
        
        # Defect type filter
        if self.defect_types:
            sample_defects = metadata.get('defect_types', [])
            if not any(defect in self.defect_types for defect in sample_defects):
                return False
        
        return True
    
    def _balance_classes(self) -> List[Dict[str, Any]]:
        """Balance classes by defect presence."""
        defect_samples = []
        no_defect_samples = []
        
        for sample in self.samples:
            if sample['metadata'].get('num_defects', 0) > 0:
                defect_samples.append(sample)
            else:
                no_defect_samples.append(sample)
        
        # Balance classes
        min_samples = min(len(defect_samples), len(no_defect_samples))
        
        if len(defect_samples) > min_samples:
            defect_samples = random.sample(defect_samples, min_samples)
        if len(no_defect_samples) > min_samples:
            no_defect_samples = random.sample(no_defect_samples, min_samples)
        
        balanced_samples = defect_samples + no_defect_samples
        random.shuffle(balanced_samples)
        
        logger.info(f"Balanced classes: {len(defect_samples)} defect samples, {len(no_defect_samples)} no-defect samples")
        
        return balanced_samples
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with caching support."""
        if self.cache_data and image_path in self.cached_images:
            return self.cached_images[image_path]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        if self.cache_data:
            self.cached_images[image_path] = image
        
        return image
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load mask with caching support."""
        if self.cache_data and mask_path in self.cached_masks:
            return self.cached_masks[mask_path]
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        if self.cache_data:
            self.cached_masks[mask_path] = mask
        
        return mask
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        sample = self.samples[idx]
        
        # Load image and mask
        image = self._load_image(sample['image_path'])
        mask = self._load_mask(sample['mask_path'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if self.target_transform:
            transformed = self.target_transform(image=mask)
            mask = transformed['image']
        
        # Ensure proper tensor format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'mask': mask,
            'metadata': sample['metadata']
        }
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        return self.samples[idx]['metadata']
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        defect_count = 0
        no_defect_count = 0
        
        for sample in self.samples:
            if sample['metadata'].get('num_defects', 0) > 0:
                defect_count += 1
            else:
                no_defect_count += 1
        
        return {
            'defect': defect_count,
            'no_defect': no_defect_count,
            'total': len(self.samples)
        }
    
    def get_modality_distribution(self) -> Dict[str, int]:
        """Get modality distribution statistics."""
        modality_counts = {}
        
        for sample in self.samples:
            modality = sample['metadata'].get('modality', 'unknown')
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return modality_counts
    
    def get_environment_distribution(self) -> Dict[str, int]:
        """Get environment distribution statistics."""
        environment_counts = {}
        
        for sample in self.samples:
            environment = sample['metadata'].get('environment', 'unknown')
            environment_counts[environment] = environment_counts.get(environment, 0) + 1
        
        return environment_counts

def create_train_transforms(config) -> A.Compose:
    """Create training data augmentation transforms."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        ToTensorV2(),
    ])

def create_val_transforms(config) -> A.Compose:
    """Create validation data transforms."""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        ToTensorV2(),
    ])

def create_target_transforms(config) -> A.Compose:
    """Create target (mask) transforms."""
    return A.Compose([
        ToTensorV2(),
    ])

def create_data_loaders(
    config,
    train_dataset: OpticalDiagnosticsDataset,
    val_dataset: OpticalDiagnosticsDataset,
    test_dataset: Optional[OpticalDiagnosticsDataset] = None
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            drop_last=False
        )
        loaders['test'] = test_loader
    
    return loaders

class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for combining different imaging modalities.
    """
    
    def __init__(
        self,
        datasets: Dict[str, OpticalDiagnosticsDataset],
        fusion_strategy: str = "concat"
    ):
        self.datasets = datasets
        self.fusion_strategy = fusion_strategy
        
        # Ensure all datasets have the same length
        lengths = [len(dataset) for dataset in datasets.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All datasets must have the same length")
        
        self.length = lengths[0]
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-modal sample."""
        samples = {}
        
        for modality, dataset in self.datasets.items():
            sample = dataset[idx]
            samples[modality] = sample
        
        # Fuse modalities
        if self.fusion_strategy == "concat":
            # Concatenate along channel dimension
            images = [sample['image'] for sample in samples.values()]
            fused_image = torch.cat(images, dim=0)
            
            # Use first modality's mask
            fused_mask = list(samples.values())[0]['mask']
            
        elif self.fusion_strategy == "average":
            # Average across modalities
            images = [sample['image'] for sample in samples.values()]
            fused_image = torch.stack(images).mean(dim=0)
            
            # Use first modality's mask
            fused_mask = list(samples.values())[0]['mask']
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return {
            'image': fused_image,
            'mask': fused_mask,
            'modalities': samples
        }

class WeightedDataset(Dataset):
    """
    Dataset with weighted sampling for handling class imbalance.
    """
    
    def __init__(self, dataset: OpticalDiagnosticsDataset, weights: Optional[List[float]] = None):
        self.dataset = dataset
        self.weights = weights
        
        if weights is None:
            # Calculate weights based on class distribution
            class_dist = dataset.get_class_distribution()
            total = class_dist['total']
            
            self.weights = []
            for i in range(len(dataset)):
                sample = dataset.get_sample_metadata(i)
                if sample.get('num_defects', 0) > 0:
                    self.weights.append(total / class_dist['defect'])
                else:
                    self.weights.append(total / class_dist['no_defect'])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]
    
    def get_weights(self) -> List[float]:
        return self.weights