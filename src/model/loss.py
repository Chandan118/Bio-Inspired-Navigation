"""
Loss functions for optical diagnostics defect detection.
Implements various loss functions optimized for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    
    Dice coefficient measures the overlap between predicted and ground truth masks.
    Dice loss = 1 - Dice coefficient
    """
    
    def __init__(self, smooth: float = 1e-6, square_denominator: bool = False, with_logits: bool = True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.with_logits = with_logits
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Dice loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # Flatten tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection
        intersection = (input_flat * target_flat).sum()
        
        # Calculate union
        if self.square_denominator:
            union = (input_flat * input_flat).sum() + (target_flat * target_flat).sum()
        else:
            union = input_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice

class BCELoss(nn.Module):
    """
    Binary Cross Entropy loss with optional focal weighting.
    """
    
    def __init__(self, with_logits: bool = True, reduction: str = 'mean'):
        super(BCELoss, self).__init__()
        self.with_logits = with_logits
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate BCE loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            BCE loss
        """
        if self.with_logits:
            return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        else:
            return F.binary_cross_entropy(input, target, reduction=self.reduction)

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, with_logits: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Focal loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')
        
        # Focal term
        pt = input * target + (1 - input) * (1 - target)
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_term = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Combine terms
        focal_loss = alpha_term * focal_term * bce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple loss types.
    
    Typically combines Dice loss and BCE loss for better training stability.
    """
    
    def __init__(
        self,
        loss_types: str = "dice_bce",
        weights: Optional[Dict[str, float]] = None,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super(CombinedLoss, self).__init__()
        
        self.loss_types = loss_types.split('_')
        self.weights = weights or {loss_type: 1.0 / len(self.loss_types) for loss_type in self.loss_types}
        
        # Initialize loss functions
        self.loss_functions = {}
        
        if 'dice' in self.loss_types:
            self.loss_functions['dice'] = DiceLoss(smooth=dice_smooth)
        
        if 'bce' in self.loss_types:
            self.loss_functions['bce'] = BCELoss()
        
        if 'focal' in self.loss_types:
            self.loss_functions['focal'] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        logger.info(f"CombinedLoss initialized with types: {self.loss_types}")
        logger.info(f"Weights: {self.weights}")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for loss_type in self.loss_types:
            if loss_type in self.loss_functions:
                loss_value = self.loss_functions[loss_type](input, target)
                losses[f'{loss_type}_loss'] = loss_value
                total_loss += self.weights[loss_type] * loss_value
        
        losses['total_loss'] = total_loss
        
        return losses

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) loss for segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6, with_logits: bool = True):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.with_logits = with_logits
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            IoU loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # Flatten tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection
        intersection = (input_flat * target_flat).sum()
        
        # Calculate union
        union = input_flat.sum() + target_flat.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return IoU loss
        return 1.0 - iou

class BoundaryLoss(nn.Module):
    """
    Boundary loss for better edge detection in segmentation.
    """
    
    def __init__(self, kernel_size: int = 3, with_logits: bool = True):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size
        self.with_logits = with_logits
        
        # Create Sobel kernels for edge detection
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32))
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate boundary loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Boundary loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        batch_size, channels, height, width = input.shape
        
        # Calculate gradients for input
        input_grad_x = F.conv2d(input, self.sobel_x.view(1, 1, 3, 3), padding=1)
        input_grad_y = F.conv2d(input, self.sobel_y.view(1, 1, 3, 3), padding=1)
        input_grad = torch.sqrt(input_grad_x**2 + input_grad_y**2)
        
        # Calculate gradients for target
        target_grad_x = F.conv2d(target, self.sobel_x.view(1, 1, 3, 3), padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y.view(1, 1, 3, 3), padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Calculate boundary loss
        boundary_loss = F.mse_loss(input_grad, target_grad)
        
        return boundary_loss

class HausdorffLoss(nn.Module):
    """
    Hausdorff distance loss for better shape matching.
    """
    
    def __init__(self, alpha: float = 2.0, with_logits: bool = True):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.with_logits = with_logits
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Hausdorff distance loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Hausdorff loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # Convert to binary masks
        input_binary = (input > 0.5).float()
        target_binary = (target > 0.5).float()
        
        # Calculate distance transforms
        input_dist = self._distance_transform(input_binary)
        target_dist = self._distance_transform(target_binary)
        
        # Calculate Hausdorff loss
        hausdorff_loss = torch.mean(torch.abs(input_dist - target_dist) ** self.alpha)
        
        return hausdorff_loss
    
    def _distance_transform(self, binary_mask: torch.Tensor) -> torch.Tensor:
        """Calculate distance transform for binary mask."""
        # This is a simplified version - in practice, you might want to use
        # a more sophisticated distance transform implementation
        batch_size, channels, height, width = binary_mask.shape
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=binary_mask.device),
            torch.arange(width, device=binary_mask.device)
        )
        
        # Find boundary pixels
        kernel = torch.ones(1, 1, 3, 3, device=binary_mask.device)
        dilated = F.conv2d(binary_mask, kernel, padding=1)
        eroded = F.conv2d(binary_mask, kernel, padding=1)
        boundary = dilated - eroded
        
        # Calculate distance to boundary
        distance_map = torch.zeros_like(binary_mask)
        
        for b in range(batch_size):
            for c in range(channels):
                mask = binary_mask[b, c]
                bound = boundary[b, c]
                
                if bound.sum() > 0:
                    # Find boundary coordinates
                    bound_coords = torch.nonzero(bound)
                    
                    # Calculate distance from each pixel to boundary
                    for i in range(height):
                        for j in range(width):
                            if mask[i, j] > 0:
                                distances = torch.sqrt(
                                    (bound_coords[:, 0] - i) ** 2 + 
                                    (bound_coords[:, 1] - j) ** 2
                                )
                                distance_map[b, c, i, j] = distances.min()
        
        return distance_map

class WeightedLoss(nn.Module):
    """
    Weighted loss wrapper for handling class imbalance.
    """
    
    def __init__(self, base_loss: nn.Module, pos_weight: float = 1.0):
        super(WeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.pos_weight = pos_weight
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted loss.
        
        Args:
            input: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
        
        Returns:
            Weighted loss
        """
        # Calculate base loss
        base_loss_value = self.base_loss(input, target)
        
        # Apply positive class weighting
        if hasattr(self.base_loss, 'with_logits') and self.base_loss.with_logits:
            input_sigmoid = torch.sigmoid(input)
        else:
            input_sigmoid = input
        
        # Calculate positive class weight
        pos_mask = (target > 0.5).float()
        neg_mask = (target <= 0.5).float()
        
        weighted_loss = (
            self.pos_weight * base_loss_value * pos_mask.mean() +
            base_loss_value * neg_mask.mean()
        )
        
        return weighted_loss

def create_loss_function(config) -> nn.Module:
    """Create loss function from configuration."""
    loss_type = config.model.loss_type
    
    if loss_type == "dice":
        return DiceLoss(smooth=config.model.dice_smooth)
    
    elif loss_type == "bce":
        return BCELoss()
    
    elif loss_type == "focal":
        return FocalLoss(
            alpha=config.model.focal_alpha,
            gamma=config.model.focal_gamma
        )
    
    elif loss_type == "dice_bce":
        return CombinedLoss(
            loss_types="dice_bce",
            weights={"dice": 0.5, "bce": 0.5},
            dice_smooth=config.model.dice_smooth
        )
    
    elif loss_type == "dice_focal":
        return CombinedLoss(
            loss_types="dice_focal",
            weights={"dice": 0.5, "focal": 0.5},
            dice_smooth=config.model.dice_smooth,
            focal_alpha=config.model.focal_alpha,
            focal_gamma=config.model.focal_gamma
        )
    
    elif loss_type == "iou":
        return IoULoss()
    
    elif loss_type == "boundary":
        return BoundaryLoss()
    
    elif loss_type == "hausdorff":
        return HausdorffLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate various segmentation metrics.
    
    Args:
        predictions: Predicted probabilities (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        threshold: Threshold for binary conversion
    
    Returns:
        Dictionary of metrics
    """
    # Convert to binary
    pred_binary = (predictions > threshold).float()
    target_binary = targets.float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # Calculate metrics
    dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    
    # Calculate precision and recall
    tp = intersection
    fp = pred_binary.sum() - intersection
    fn = target_binary.sum() - intersection
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }