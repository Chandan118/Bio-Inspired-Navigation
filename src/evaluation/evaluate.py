"""
Evaluation module for AutoOpticalDiagnostics.
Provides comprehensive model evaluation with multiple metrics and visualizations.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import cv2
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from ..config import Config
from ..model.unet_model import create_model
from ..model.dataset import OpticalDataset, get_validation_transforms
from ..utils import setup_logging, create_visualization_grid, format_time


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    batch_size: int = 8
    num_workers: int = 4
    device: str = "auto"
    threshold: float = 0.5
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    threshold_steps: int = 50
    save_predictions: bool = True
    save_attention_maps: bool = True
    save_gradcam: bool = True
    num_visualization_samples: int = 20
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95, 0.99])
    bootstrap_samples: int = 1000
    metrics: List[str] = field(default_factory=lambda: [
        "dice_score", "iou", "precision", "recall", "f1_score",
        "hausdorff_distance", "surface_dice", "psnr", "ssim"
    ])


class EvaluationMetrics:
    """Comprehensive evaluation metrics for segmentation tasks."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.hausdorff_distances = []
        self.surface_dice_scores = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.predictions = []
        self.targets = []
        self.confidence_scores = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, confidence: Optional[torch.Tensor] = None):
        """Update metrics with new predictions and targets."""
        pred_binary = (pred > self.threshold).float()
        
        # Convert to numpy for metric calculation
        pred_np = pred_binary.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_raw_np = pred.cpu().numpy()
        
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_mask = pred_np[i, 0]  # Remove channel dimension
            target_mask = target_np[i, 0]
            pred_raw = pred_raw_np[i, 0]
            
            # Calculate metrics
            dice = self._calculate_dice_score(pred_mask, target_mask)
            iou = self._calculate_iou(pred_mask, target_mask)
            precision, recall, f1 = self._calculate_precision_recall_f1(pred_mask, target_mask)
            hausdorff = self._calculate_hausdorff_distance(pred_mask, target_mask)
            surface_dice = self._calculate_surface_dice(pred_mask, target_mask)
            psnr = self._calculate_psnr(pred_raw, target_mask)
            ssim = self._calculate_ssim(pred_raw, target_mask)
            
            # Store metrics
            self.dice_scores.append(dice)
            self.iou_scores.append(iou)
            self.precision_scores.append(precision)
            self.recall_scores.append(recall)
            self.f1_scores.append(f1)
            self.hausdorff_distances.append(hausdorff)
            self.surface_dice_scores.append(surface_dice)
            self.psnr_scores.append(psnr)
            self.ssim_scores.append(ssim)
            
            # Store raw predictions and targets for ROC/PR curves
            self.predictions.extend(pred_raw.flatten())
            self.targets.extend(target_mask.flatten())
            
            if confidence is not None:
                self.confidence_scores.extend(confidence[i].cpu().numpy().flatten())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {
            'dice_score': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'precision': np.mean(self.precision_scores),
            'recall': np.mean(self.recall_scores),
            'f1_score': np.mean(self.f1_scores),
            'hausdorff_distance': np.mean(self.hausdorff_distances),
            'surface_dice': np.mean(self.surface_dice_scores),
            'psnr': np.mean(self.psnr_scores),
            'ssim': np.mean(self.ssim_scores)
        }
        
        # Add standard deviations
        metrics.update({
            'dice_score_std': np.std(self.dice_scores),
            'iou_std': np.std(self.iou_scores),
            'precision_std': np.std(self.precision_scores),
            'recall_std': np.std(self.recall_scores),
            'f1_score_std': np.std(self.f1_scores),
            'hausdorff_distance_std': np.std(self.hausdorff_distances),
            'surface_dice_std': np.std(self.surface_dice_scores),
            'psnr_std': np.std(self.psnr_scores),
            'ssim_std': np.std(self.ssim_scores)
        })
        
        return metrics
    
    def _calculate_dice_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Dice score."""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        return (2.0 * intersection) / (union + 1e-8)
    
    def _calculate_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return intersection / (union + 1e-8)
    
    def _calculate_precision_recall_f1(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1-score."""
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        fn = np.sum((1 - pred) * target)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Hausdorff distance."""
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return float('inf')
        
        # Find contours
        pred_contours = self._find_contours(pred)
        target_contours = self._find_contours(target)
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            return float('inf')
        
        # Calculate Hausdorff distance
        min_distances = []
        for pred_contour in pred_contours:
            distances = []
            for target_contour in target_contours:
                dist = directed_hausdorff(pred_contour, target_contour)[0]
                distances.append(dist)
            min_distances.append(min(distances))
        
        return np.mean(min_distances) if min_distances else float('inf')
    
    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours in binary mask."""
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        return [contour.squeeze() for contour in contours if len(contour) > 2]
    
    def _calculate_surface_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate surface Dice score."""
        pred_surface = self._extract_surface(pred)
        target_surface = self._extract_surface(target)
        
        intersection = np.sum(pred_surface * target_surface)
        union = np.sum(pred_surface) + np.sum(target_surface)
        return (2.0 * intersection) / (union + 1e-8)
    
    def _extract_surface(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface from binary mask."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        surface = mask.astype(np.uint8) - eroded
        return surface
    
    def _calculate_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def _calculate_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Simplified SSIM calculation
        mu_x = np.mean(pred)
        mu_y = np.mean(target)
        sigma_x = np.std(pred)
        sigma_y = np.std(target)
        sigma_xy = np.mean((pred - mu_x) * (target - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        
        return ssim


class Evaluator:
    """Comprehensive model evaluator with visualization capabilities."""
    
    def __init__(self, config: EvaluationConfig, model_path: str, data_dir: str):
        self.config = config
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.device = self._get_device()
        self.logger = setup_logging()
        
        # Load model
        self.model = self._load_model()
        
        # Setup data loader
        self.val_loader = self._setup_data_loader()
        
        # Setup output directories
        self.output_dir = Path("outputs/evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = EvaluationMetrics(threshold=config.threshold)
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            model_config = {}
        
        # Create model
        model = create_model(
            model_type=model_config.get('model_type', 'unet'),
            n_channels=model_config.get('input_channels', 3),
            n_classes=model_config.get('output_channels', 1),
            initial_features=model_config.get('initial_features', 64),
            depth=model_config.get('depth', 5)
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded successfully on {self.device}")
        return model
    
    def _setup_data_loader(self) -> DataLoader:
        """Setup validation data loader."""
        val_images_dir = self.data_dir / "val" / "images"
        val_masks_dir = self.data_dir / "val" / "masks"
        
        if not val_images_dir.exists() or not val_masks_dir.exists():
            raise FileNotFoundError(f"Validation data not found in {self.data_dir}")
        
        # Create dataset
        transforms = get_validation_transforms()
        dataset = OpticalDataset(
            images_dir=str(val_images_dir),
            masks_dir=str(val_masks_dir),
            transform=transforms
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Validation data loader created with {len(dataset)} samples")
        return loader
    
    def evaluate(self) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        self.logger.info("Starting model evaluation...")
        start_time = time.time()
        
        # Reset metrics
        self.metrics.reset()
        
        # Evaluation loop
        predictions_list = []
        targets_list = []
        images_list = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                predictions = torch.sigmoid(predictions)
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Store for visualization
                if batch_idx < self.config.num_visualization_samples // self.config.batch_size:
                    predictions_list.append(predictions.cpu())
                    targets_list.append(targets.cpu())
                    images_list.append(images.cpu())
        
        # Compute final metrics
        metrics = self.metrics.compute()
        
        # Generate visualizations
        if predictions_list:
            self._generate_visualizations(
                torch.cat(predictions_list, dim=0),
                torch.cat(targets_list, dim=0),
                torch.cat(images_list, dim=0)
            )
        
        # Generate evaluation report
        report = self._generate_evaluation_report(metrics)
        
        # Save results
        self._save_results(metrics, report)
        
        evaluation_time = time.time() - start_time
        self.logger.info(f"Evaluation completed in {format_time(evaluation_time)}")
        
        return {
            'metrics': metrics,
            'report': report,
            'evaluation_time': evaluation_time
        }
    
    def _generate_visualizations(self, predictions: torch.Tensor, targets: torch.Tensor, images: torch.Tensor):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations...")
        
        # Create sample predictions grid
        self._create_prediction_grid(predictions, targets, images)
        
        # Create ROC and PR curves
        self._create_roc_pr_curves()
        
        # Create confusion matrix
        self._create_confusion_matrix()
        
        # Create metrics distribution plots
        self._create_metrics_distributions()
        
        # Create threshold analysis
        self._create_threshold_analysis()
    
    def _create_prediction_grid(self, predictions: torch.Tensor, targets: torch.Tensor, images: torch.Tensor):
        """Create prediction visualization grid."""
        num_samples = min(self.config.num_visualization_samples, len(predictions))
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            img = images[i].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # Ground truth
            target = targets[i, 0].numpy()
            axes[i, 1].imshow(target, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Prediction
            pred = predictions[i, 0].numpy()
            axes[i, 2].imshow(pred, cmap='viridis')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            # Binary prediction
            pred_binary = (pred > self.config.threshold).astype(np.float32)
            axes[i, 3].imshow(pred_binary, cmap='gray')
            axes[i, 3].set_title(f"Binary (threshold={self.config.threshold})")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roc_pr_curves(self):
        """Create ROC and Precision-Recall curves."""
        if not self.metrics.predictions or not self.metrics.targets:
            return
        
        predictions = np.array(self.metrics.predictions)
        targets = np.array(self.metrics.targets)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(targets, predictions)
        pr_auc = auc(recall, precision)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # PR curve
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_pr_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrix(self):
        """Create confusion matrix."""
        if not self.metrics.predictions or not self.metrics.targets:
            return
        
        predictions = np.array(self.metrics.predictions)
        targets = np.array(self.metrics.targets)
        
        # Convert to binary
        pred_binary = (predictions > self.config.threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, pred_binary)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_distributions(self):
        """Create distribution plots for metrics."""
        metrics_data = {
            'Dice Score': self.metrics.dice_scores,
            'IoU': self.metrics.iou_scores,
            'Precision': self.metrics.precision_scores,
            'Recall': self.metrics.recall_scores,
            'F1 Score': self.metrics.f1_scores,
            'Hausdorff Distance': self.metrics.hausdorff_distances,
            'Surface Dice': self.metrics.surface_dice_scores
        }
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i >= len(axes):
                break
            
            # Filter out infinite values
            values = [v for v in values if not np.isinf(v)]
            
            if values:
                axes[i].hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(values):.3f}')
                axes[i].set_title(f'{metric_name} Distribution')
                axes[i].set_xlabel(metric_name)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_threshold_analysis(self):
        """Create threshold analysis plots."""
        if not self.metrics.predictions or not self.metrics.targets:
            return
        
        predictions = np.array(self.metrics.predictions)
        targets = np.array(self.metrics.targets)
        
        thresholds = np.linspace(self.config.threshold_range[0], 
                               self.config.threshold_range[1], 
                               self.config.threshold_steps)
        
        metrics_vs_threshold = {
            'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        for threshold in thresholds:
            pred_binary = (predictions > threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum(pred_binary * targets)
            fp = np.sum(pred_binary * (1 - targets))
            fn = np.sum((1 - pred_binary) * targets)
            tn = np.sum((1 - pred_binary) * (1 - targets))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            intersection = tp
            union = tp + fp + fn
            iou = intersection / (union + 1e-8)
            
            dice = 2 * intersection / (2 * intersection + fp + fn + 1e-8)
            
            metrics_vs_threshold['dice'].append(dice)
            metrics_vs_threshold['iou'].append(iou)
            metrics_vs_threshold['precision'].append(precision)
            metrics_vs_threshold['recall'].append(recall)
            metrics_vs_threshold['f1'].append(f1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in metrics_vs_threshold.items():
            ax.plot(thresholds, values, label=metric_name.capitalize(), linewidth=2)
        
        ax.axvline(self.config.threshold, color='red', linestyle='--', 
                   label=f'Current threshold ({self.config.threshold})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_evaluation_report(self, metrics: Dict[str, float]) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Model Path: {self.model_path}")
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append(f"Threshold: {self.config.threshold}")
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION:")
        report.append("-" * 20)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        report.append(f"Total Parameters: {total_params:,}")
        report.append(f"Trainable Parameters: {trainable_params:,}")
        report.append("")
        
        # Metrics summary
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 20)
        
        metric_names = {
            'dice_score': 'Dice Score',
            'iou': 'IoU',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'hausdorff_distance': 'Hausdorff Distance',
            'surface_dice': 'Surface Dice',
            'psnr': 'PSNR',
            'ssim': 'SSIM'
        }
        
        for metric_key, metric_name in metric_names.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                std_key = f"{metric_key}_std"
                if std_key in metrics:
                    std = metrics[std_key]
                    report.append(f"{metric_name}: {value:.4f} ± {std:.4f}")
                else:
                    report.append(f"{metric_name}: {value:.4f}")
        
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION:")
        report.append("-" * 20)
        report.append(f"Validation Samples: {len(self.val_loader.dataset)}")
        report.append(f"Batch Size: {self.config.batch_size}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if metrics.get('dice_score', 0) < 0.7:
            report.append("- Consider improving model architecture or training data")
        if metrics.get('hausdorff_distance', float('inf')) > 10:
            report.append("- Model struggles with boundary accuracy, consider boundary-aware loss")
        if metrics.get('precision', 0) < 0.8:
            report.append("- High false positive rate, consider adjusting threshold")
        if metrics.get('recall', 0) < 0.8:
            report.append("- High false negative rate, consider data augmentation")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _save_results(self, metrics: Dict[str, float], report: str):
        """Save evaluation results."""
        # Save metrics as JSON
        metrics_file = self.output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save report as text
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Results saved to {self.output_dir}")


def create_evaluation_config(
    batch_size: int = 8,
    threshold: float = 0.5,
    num_visualization_samples: int = 20,
    save_predictions: bool = True
) -> EvaluationConfig:
    """Create evaluation configuration."""
    return EvaluationConfig(
        batch_size=batch_size,
        threshold=threshold,
        num_visualization_samples=num_visualization_samples,
        save_predictions=save_predictions
    )