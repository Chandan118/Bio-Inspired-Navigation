"""
Loss functions for optical defect detection segmentation.
Includes Dice, BCE, Focal, and combined loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        square_denominator: bool = False,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor
            square_denominator: Whether to square the denominator
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.with_logits = with_logits
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Dice loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        flat_input = input.view(-1)
        flat_target = target.view(-1)
        
        intersection = (flat_input * flat_target).sum()
        
        if self.square_denominator:
            denominator = (flat_input * flat_input).sum() + (flat_target * flat_target).sum()
        else:
            denominator = flat_input.sum() + flat_target.sum()
        
        loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth))
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
        square_denominator: bool = False,
        reduction: str = "mean"
    ):
        """
        Initialize Dice BCE Loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor
            square_denominator: Whether to square the denominator
            reduction: Reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth, square_denominator, True, reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Combined loss
        """
        dice_loss = self.dice_loss(input, target)
        bce_loss = self.bce_loss(input, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Alpha parameter for class balancing
            gamma: Gamma parameter for focusing
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        focal_weight: float = 0.2,
        smooth: float = 1e-6,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
            smooth: Smoothing factor
            alpha: Alpha parameter for Focal loss
            gamma: Gamma parameter for Focal loss
            reduction: Reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=smooth, with_logits=True, reduction=reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Combined loss
        """
        dice_loss = self.dice_loss(input, target)
        bce_loss = self.bce_loss(input, target)
        focal_loss = self.focal_loss(input, target)
        
        total_loss = (
            self.dice_weight * dice_loss +
            self.bce_weight * bce_loss +
            self.focal_weight * focal_loss
        )
        
        return total_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize IoU Loss.
        
        Args:
            smooth: Smoothing factor
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.with_logits = with_logits
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            IoU loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        flat_input = input.view(-1)
        flat_target = target.view(-1)
        
        intersection = (flat_input * flat_target).sum()
        union = flat_input.sum() + flat_target.sum() - intersection
        
        loss = 1 - ((intersection + self.smooth) / (union + self.smooth))
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.with_logits = with_logits
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Tversky loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        flat_input = input.view(-1)
        flat_target = target.view(-1)
        
        intersection = (flat_input * flat_target).sum()
        fps = (flat_input * (1 - flat_target)).sum()
        fns = ((1 - flat_input) * flat_target).sum()
        
        denominator = intersection + self.alpha * fps + self.beta * fns
        
        loss = 1 - ((intersection + self.smooth) / (denominator + self.smooth))
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better boundary prediction.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize Boundary Loss.
        
        Args:
            kernel_size: Kernel size for boundary detection
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.with_logits = with_logits
        self.reduction = reduction
        
        # Create boundary detection kernel
        self.boundary_kernel = self._create_boundary_kernel()
    
    def _create_boundary_kernel(self) -> torch.Tensor:
        """Create boundary detection kernel."""
        kernel = torch.ones(self.kernel_size, self.kernel_size)
        kernel[self.kernel_size//2, self.kernel_size//2] = -(self.kernel_size * self.kernel_size - 1)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Boundary loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # Detect boundaries
        input_boundary = F.conv2d(input, self.boundary_kernel.to(input.device), padding=self.kernel_size//2)
        target_boundary = F.conv2d(target, self.boundary_kernel.to(target.device), padding=self.kernel_size//2)
        
        # Calculate boundary loss
        boundary_loss = F.mse_loss(input_boundary, target_boundary, reduction=self.reduction)
        
        return boundary_loss


class HausdorffLoss(nn.Module):
    """
    Hausdorff Distance Loss approximation.
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize Hausdorff Loss.
        
        Args:
            alpha: Alpha parameter
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.with_logits = with_logits
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Hausdorff loss
        """
        if self.with_logits:
            input = torch.sigmoid(input)
        
        # Calculate distance transform
        input_dist = self._distance_transform(input)
        target_dist = self._distance_transform(target)
        
        # Calculate Hausdorff loss
        hausdorff_loss = torch.abs(input_dist - target_dist) ** self.alpha
        
        if self.reduction == "mean":
            return hausdorff_loss.mean()
        elif self.reduction == "sum":
            return hausdorff_loss.sum()
        else:
            return hausdorff_loss
    
    def _distance_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate distance transform."""
        # Simple approximation using morphological operations
        kernel = torch.ones(3, 3).unsqueeze(0).unsqueeze(0).to(x.device)
        
        # Erosion
        eroded = F.conv2d(x, kernel, padding=1)
        eroded = (eroded > 0.5).float()
        
        # Distance approximation
        distance = 1 - eroded
        
        return distance


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that automatically adjusts weights based on training progress.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float] = None,
        adaptation_rate: float = 0.01,
        with_logits: bool = True,
        reduction: str = "mean"
    ):
        """
        Initialize Adaptive Loss.
        
        Args:
            initial_weights: Initial weights for different loss components
            adaptation_rate: Rate of weight adaptation
            with_logits: Whether input is logits
            reduction: Reduction method
        """
        super().__init__()
        self.adaptation_rate = adaptation_rate
        self.with_logits = with_logits
        self.reduction = reduction
        
        if initial_weights is None:
            initial_weights = {
                'dice': 0.4,
                'bce': 0.3,
                'focal': 0.2,
                'iou': 0.1
            }
        
        self.weights = nn.Parameter(torch.tensor(list(initial_weights.values())))
        self.loss_functions = {
            'dice': DiceLoss(with_logits=with_logits, reduction=reduction),
            'bce': nn.BCEWithLogitsLoss(reduction=reduction),
            'focal': FocalLoss(reduction=reduction),
            'iou': IoULoss(with_logits=with_logits, reduction=reduction)
        }
        
        self.loss_history = {name: [] for name in initial_weights.keys()}
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Predicted logits
            target: Ground truth masks
            
        Returns:
            Adaptive loss
        """
        # Calculate individual losses
        losses = {}
        for name, loss_fn in self.loss_functions.items():
            losses[name] = loss_fn(input, target)
            self.loss_history[name].append(losses[name].item())
        
        # Normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # Calculate weighted loss
        total_loss = sum(weights[i] * losses[name] for i, name in enumerate(self.loss_functions.keys()))
        
        return total_loss
    
    def update_weights(self):
        """Update weights based on loss history."""
        if len(self.loss_history[list(self.loss_functions.keys())[0]]) < 10:
            return
        
        # Calculate average losses
        avg_losses = {}
        for name in self.loss_functions.keys():
            recent_losses = self.loss_history[name][-10:]
            avg_losses[name] = np.mean(recent_losses)
        
        # Update weights based on loss performance
        total_loss = sum(avg_losses.values())
        new_weights = []
        
        for name in self.loss_functions.keys():
            # Lower loss should get higher weight
            weight = 1.0 / (avg_losses[name] + 1e-8)
            new_weights.append(weight)
        
        # Normalize and update
        new_weights = torch.tensor(new_weights, device=self.weights.device)
        new_weights = F.softmax(new_weights, dim=0)
        
        with torch.no_grad():
            self.weights.data = (
                (1 - self.adaptation_rate) * self.weights.data +
                self.adaptation_rate * new_weights
            )


def create_loss_function(
    loss_type: str = "dice_bce",
    **kwargs
) -> nn.Module:
    """
    Create loss function based on type.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    if loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "dice_bce":
        return DiceBCELoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "iou":
        return IoULoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "boundary":
        return BoundaryLoss(**kwargs)
    elif loss_type == "hausdorff":
        return HausdorffLoss(**kwargs)
    elif loss_type == "adaptive":
        return AdaptiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")