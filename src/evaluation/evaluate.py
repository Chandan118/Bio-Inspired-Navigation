"""
Evaluation module for AutoOpticalDiagnostics.
Implements comprehensive model evaluation and analysis.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from scipy import stats
import pandas as pd

from ..model import create_unet_model, OpticalDiagnosticsDataset, create_val_transforms
from ..utils import (
    setup_device, load_model_checkpoint, calculate_metrics, visualize_predictions,
    create_performance_report, create_interactive_plots, calculate_statistical_significance
)
from ..config import get_config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluator for optical diagnostics.
    
    Features:
    - Multi-metric evaluation
    - Statistical analysis
    - Visualization generation
    - Performance comparison
    - Threshold optimization
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = setup_device(self.config.training.device)
        self.model = None
        self.test_loader = None
        
        logger.info(f"ModelEvaluator initialized on device: {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model."""
        # Create model
        self.model = create_unet_model(self.config)
        self.model = self.model.to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def setup_test_data(self, data_dir: Path) -> None:
        """Setup test data loader."""
        # Create transforms
        test_transforms = create_val_transforms(self.config)
        
        # Create dataset
        test_dataset = OpticalDiagnosticsDataset(
            data_dir=data_dir,
            split="val",  # Using validation split for testing
            transform=test_transforms,
            modalities=self.config.data_generation.materials
        )
        
        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def evaluate_model(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluate model performance."""
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and test data must be loaded first")
        
        all_predictions = []
        all_targets = []
        all_images = []
        all_metrics = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating model"):
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # Calculate metrics
                batch_metrics = calculate_metrics(predictions, masks, threshold)
                all_metrics.append(batch_metrics)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                all_images.append(images.cpu().numpy())
        
        # Concatenate results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_predictions, all_targets, threshold)
        
        # Calculate per-sample metrics
        per_sample_metrics = self._calculate_per_sample_metrics(all_predictions, all_targets, threshold)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(per_sample_metrics)
        
        return {
            'overall_metrics': overall_metrics,
            'per_sample_metrics': per_sample_metrics,
            'statistical_analysis': statistical_analysis,
            'predictions': all_predictions,
            'targets': all_targets,
            'images': all_images
        }
    
    def _calculate_overall_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                 threshold: float) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        # Convert to binary
        pred_binary = (predictions > threshold).astype(np.uint8)
        target_binary = targets.astype(np.uint8)
        
        # Flatten for calculation
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate IoU
        intersection = np.sum((pred_binary == 1) & (target_binary == 1))
        union = np.sum((pred_binary == 1) | (target_binary == 1))
        iou = intersection / union if union > 0 else 0.0
        
        # Calculate Dice
        dice = 2 * intersection / (np.sum(pred_binary) + np.sum(target_binary)) if (np.sum(pred_binary) + np.sum(target_binary)) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'dice_score': dice,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
    
    def _calculate_per_sample_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                    threshold: float) -> List[Dict[str, float]]:
        """Calculate metrics for each sample."""
        per_sample_metrics = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            target = targets[i]
            
            # Calculate sample metrics
            sample_metrics = calculate_metrics(
                torch.from_numpy(pred), 
                torch.from_numpy(target), 
                threshold
            )
            
            per_sample_metrics.append(sample_metrics)
        
        return per_sample_metrics
    
    def _perform_statistical_analysis(self, per_sample_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform statistical analysis on metrics."""
        # Convert to DataFrame
        df = pd.DataFrame(per_sample_metrics)
        
        # Calculate statistics
        statistics = {}
        for metric in df.columns:
            values = df[metric].values
            statistics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'confidence_interval_95': stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
            }
        
        return statistics
    
    def optimize_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Optimize classification threshold."""
        thresholds = np.linspace(
            self.config.evaluation.threshold_range[0],
            self.config.evaluation.threshold_range[1],
            self.config.evaluation.threshold_steps
        )
        
        threshold_metrics = []
        
        for threshold in tqdm(thresholds, desc="Optimizing threshold"):
            metrics = self._calculate_overall_metrics(predictions, targets, threshold)
            threshold_metrics.append({
                'threshold': threshold,
                **metrics
            })
        
        # Find best threshold for different metrics
        df = pd.DataFrame(threshold_metrics)
        
        best_thresholds = {}
        for metric in ['f1_score', 'dice_score', 'iou', 'precision', 'recall']:
            best_idx = df[metric].idxmax()
            best_thresholds[metric] = {
                'threshold': df.loc[best_idx, 'threshold'],
                'value': df.loc[best_idx, metric]
            }
        
        return {
            'threshold_metrics': threshold_metrics,
            'best_thresholds': best_thresholds,
            'recommended_threshold': best_thresholds['f1_score']['threshold']
        }
    
    def generate_visualizations(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate comprehensive visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion matrix
        self._plot_confusion_matrix(results, output_dir)
        
        # 2. ROC curve
        self._plot_roc_curve(results, output_dir)
        
        # 3. Threshold optimization
        self._plot_threshold_optimization(results, output_dir)
        
        # 4. Sample predictions
        self._plot_sample_predictions(results, output_dir)
        
        # 5. Metrics distribution
        self._plot_metrics_distribution(results, output_dir)
        
        # 6. Performance comparison
        self._plot_performance_comparison(results, output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_confusion_matrix(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot confusion matrix."""
        metrics = results['overall_metrics']
        
        cm = np.array([
            [metrics['tn'], metrics['fp']],
            [metrics['fn'], metrics['tp']]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot ROC curve."""
        predictions = results['predictions'].flatten()
        targets = results['targets'].flatten()
        
        fpr, tpr, _ = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_optimization(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot threshold optimization results."""
        if 'threshold_optimization' not in results:
            return
        
        threshold_data = results['threshold_optimization']['threshold_metrics']
        df = pd.DataFrame(threshold_data)
        
        plt.figure(figsize=(12, 8))
        
        metrics = ['f1_score', 'dice_score', 'iou', 'precision', 'recall']
        for metric in metrics:
            plt.plot(df['threshold'], df[metric], label=metric.capitalize(), linewidth=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_predictions(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot sample predictions."""
        images = results['images']
        targets = results['targets']
        predictions = results['predictions']
        
        # Select random samples
        num_samples = min(8, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        visualize_predictions(
            images[indices],
            targets[indices],
            predictions[indices],
            str(output_dir / 'sample_predictions.png'),
            num_samples=num_samples
        )
    
    def _plot_metrics_distribution(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot metrics distribution."""
        per_sample_metrics = results['per_sample_metrics']
        df = pd.DataFrame(per_sample_metrics)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['dice_score', 'iou', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].hist(df[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.capitalize()} Distribution')
                axes[i].set_xlabel(metric.capitalize())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove extra subplot
        if len(axes) > len(metrics):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Plot performance comparison."""
        overall_metrics = results['overall_metrics']
        
        metrics = ['precision', 'recall', 'f1_score', 'iou', 'dice_score']
        values = [overall_metrics[metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate comprehensive evaluation report."""
        # Create performance report
        create_performance_report(
            results['overall_metrics'],
            self.config.__dict__,
            str(output_dir / 'performance_report.txt')
        )
        
        # Save detailed results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create interactive plots
        if 'threshold_optimization' in results:
            create_interactive_plots(
                results['threshold_optimization']['threshold_metrics'],
                str(output_dir / 'interactive_plots.html')
            )
        
        logger.info(f"Evaluation report saved to {output_dir}")

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate optical diagnostics model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    evaluator.load_model(args.model_path)
    
    # Setup test data
    evaluator.setup_test_data(Path(args.data_dir))
    
    # Evaluate model
    results = evaluator.evaluate_model(threshold=args.threshold)
    
    # Optimize threshold
    threshold_results = evaluator.optimize_threshold(results['predictions'], results['targets'])
    results['threshold_optimization'] = threshold_results
    
    # Generate visualizations and report
    output_dir = Path(args.output_dir)
    evaluator.generate_visualizations(results, output_dir)
    evaluator.generate_report(results, output_dir)
    
    # Print summary
    print("Evaluation completed!")
    print(f"Best F1 Score: {threshold_results['best_thresholds']['f1_score']['value']:.4f}")
    print(f"Recommended Threshold: {threshold_results['recommended_threshold']:.3f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()