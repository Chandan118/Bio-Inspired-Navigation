"""
Utility functions for AutoOpticalDiagnostics project.
Contains helper functions for data processing, visualization, and system operations.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from datetime import datetime
import traceback

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
from PIL import Image, ImageDraw, ImageFont
import psutil
import GPUtil
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from .config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_optical_diagnostics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()

class PerformanceMonitor:
    """Monitor system performance during training and inference."""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.gpu_usage = []
        self.cpu_usage = []
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.memory_usage = []
        self.gpu_usage = []
        self.cpu_usage = []
        
    def record(self):
        """Record current system metrics."""
        if self.start_time is None:
            return
            
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append({
            'timestamp': time.time() - self.start_time,
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3)
        })
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.append({
            'timestamp': time.time() - self.start_time,
            'percent': cpu_percent
        })
        
        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if i >= len(self.gpu_usage):
                    self.gpu_usage.append([])
                self.gpu_usage[i].append({
                    'timestamp': time.time() - self.start_time,
                    'memory_percent': gpu.memoryUtil * 100,
                    'load_percent': gpu.load * 100,
                    'temperature': gpu.temperature
                })
        except:
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.memory_usage:
            return {}
            
        total_time = time.time() - self.start_time
        
        summary = {
            'total_time_seconds': total_time,
            'memory': {
                'peak_percent': max(m['percent'] for m in self.memory_usage),
                'average_percent': np.mean([m['percent'] for m in self.memory_usage]),
                'peak_used_gb': max(m['used_gb'] for m in self.memory_usage)
            },
            'cpu': {
                'peak_percent': max(c['percent'] for c in self.cpu_usage),
                'average_percent': np.mean([c['percent'] for c in self.cpu_usage])
            }
        }
        
        if self.gpu_usage:
            summary['gpu'] = {}
            for i, gpu_data in enumerate(self.gpu_usage):
                if gpu_data:
                    summary['gpu'][f'gpu_{i}'] = {
                        'peak_memory_percent': max(g['memory_percent'] for g in gpu_data),
                        'average_memory_percent': np.mean([g['memory_percent'] for g in gpu_data]),
                        'peak_load_percent': max(g['load_percent'] for g in gpu_data),
                        'average_load_percent': np.mean([g['load_percent'] for g in gpu_data),
                        'peak_temperature': max(g['temperature'] for g in gpu_data)
                    }
        
        return summary

def setup_device(device_preference: str = "auto") -> torch.device:
    """Setup the best available device for PyTorch operations."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_preference)
        logger.info(f"Using specified device: {device}")
    
    return device

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def create_directory_structure(base_path: Path) -> None:
    """Create the complete directory structure for the project."""
    directories = [
        base_path / "data" / "synthetic" / "train" / "images",
        base_path / "data" / "synthetic" / "train" / "masks",
        base_path / "data" / "synthetic" / "val" / "images",
        base_path / "data" / "synthetic" / "val" / "masks",
        base_path / "models" / "saved_models",
        base_path / "outputs" / "evaluation_results",
        base_path / "logs",
        base_path / "checkpoints",
        base_path / "tensorboard_logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint with comprehensive information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': get_config().__dict__,
        'timestamp': datetime.now().isoformat(),
        'is_best': is_best
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Tuple[int, float, Dict[str, float]]:
    """Load model checkpoint and return training state."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint['metrics']
    
    logger.info(f"Checkpoint loaded from {filepath}")
    logger.info(f"Resuming from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss, metrics

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    # Apply threshold
    pred_binary = (predictions > threshold).astype(np.uint8)
    target_binary = targets.astype(np.uint8)
    
    # Basic metrics
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # IoU (Intersection over Union)
    intersection = np.sum((pred_binary == 1) & (target_binary == 1))
    union = np.sum((pred_binary == 1) | (target_binary == 1))
    iou = intersection / union if union > 0 else 0.0
    
    # Dice coefficient
    dice = 2 * intersection / (np.sum(pred_binary) + np.sum(target_binary)) if (np.sum(pred_binary) + np.sum(target_binary)) > 0 else 0.0
    
    # Hausdorff distance
    try:
        hausdorff_dist = directed_hausdorff(pred_binary, target_binary)[0]
    except:
        hausdorff_dist = float('inf')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou,
        'dice_score': dice,
        'hausdorff_distance': hausdorff_dist,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def visualize_predictions(
    images: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    num_samples: int = 8,
    threshold: float = 0.5
) -> None:
    """Create comprehensive visualization of model predictions."""
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i].transpose(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(targets[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Raw prediction
        axes[i, 2].imshow(predictions[i], cmap='viridis')
        axes[i, 2].set_title('Raw Prediction')
        axes[i, 2].axis('off')
        
        # Thresholded prediction
        pred_binary = (predictions[i] > threshold).astype(np.uint8)
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title('Thresholded Prediction')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualization saved to {save_path}")

def create_performance_report(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: str
) -> None:
    """Create a comprehensive performance report."""
    report = f"""
# AutoOpticalDiagnostics Performance Report

## Model Configuration
- **Architecture**: U-Net with {config['model']['depth']} levels
- **Input Channels**: {config['model']['input_channels']}
- **Output Channels**: {config['model']['output_channels']}
- **Initial Features**: {config['model']['initial_features']}
- **Dropout Rate**: {config['model']['dropout_rate']}

## Training Configuration
- **Batch Size**: {config['training']['batch_size']}
- **Learning Rate**: {config['training']['learning_rate']}
- **Epochs**: {config['training']['num_epochs']}
- **Loss Function**: {config['training']['loss_type']}

## Performance Metrics
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}
- **IoU**: {metrics['iou']:.4f}
- **Dice Score**: {metrics['dice_score']:.4f}
- **Hausdorff Distance**: {metrics['hausdorff_distance']:.4f}

## Confusion Matrix
- **True Positives**: {metrics['tp']}
- **True Negatives**: {metrics['tn']}
- **False Positives**: {metrics['fp']}
- **False Negatives**: {metrics['fn']}

## Data Generation
- **Training Samples**: {config['data_generation']['num_train_samples']}
- **Validation Samples**: {config['data_generation']['num_val_samples']}
- **Image Size**: {config['data_generation']['image_size']}
- **Defect Probability**: {config['data_generation']['defect_probability']}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Performance report saved to {save_path}")

def create_interactive_plots(
    metrics_history: Dict[str, List[float]],
    save_path: str
) -> None:
    """Create interactive plots using Plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Metrics', 'Precision-Recall', 'IoU vs Dice'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training loss
    if 'train_loss' in metrics_history:
        fig.add_trace(
            go.Scatter(y=metrics_history['train_loss'], name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Validation metrics
    if 'val_loss' in metrics_history:
        fig.add_trace(
            go.Scatter(y=metrics_history['val_loss'], name='Validation Loss', line=dict(color='red')),
            row=1, col=2
        )
    
    # Precision-Recall curve
    if 'precision' in metrics_history and 'recall' in metrics_history:
        fig.add_trace(
            go.Scatter(x=metrics_history['recall'], y=metrics_history['precision'], 
                      name='Precision-Recall', mode='lines+markers'),
            row=2, col=1
        )
    
    # IoU vs Dice
    if 'iou' in metrics_history and 'dice_score' in metrics_history:
        fig.add_trace(
            go.Scatter(x=metrics_history['iou'], y=metrics_history['dice_score'], 
                      name='IoU vs Dice', mode='lines+markers'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Training Progress")
    fig.write_html(save_path)
    logger.info(f"Interactive plots saved to {save_path}")

def print_system_info() -> None:
    """Print comprehensive system information."""
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="magenta")
    
    # CPU Info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    table.add_row("CPU Cores", str(cpu_count))
    table.add_row("CPU Frequency", f"{cpu_freq.current:.1f} MHz")
    
    # Memory Info
    memory = psutil.virtual_memory()
    table.add_row("Total Memory", f"{memory.total / (1024**3):.1f} GB")
    table.add_row("Available Memory", f"{memory.available / (1024**3):.1f} GB")
    
    # GPU Info
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            table.add_row(f"GPU {i} Name", gpu.name)
            table.add_row(f"GPU {i} Memory", f"{gpu.memoryTotal} MB")
    except:
        table.add_row("GPU", "Not available")
    
    # PyTorch Info
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Count", str(torch.cuda.device_count()))
    
    console.print(table)

def create_progress_bar(description: str, total: int) -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )

def log_experiment_config(config: Dict[str, Any], experiment_name: str) -> None:
    """Log experiment configuration for reproducibility."""
    config_path = Path("logs") / f"{experiment_name}_config.json"
    
    # Convert numpy arrays and other non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return obj
    
    config_serializable = json.loads(
        json.dumps(config, default=convert_for_json)
    )
    
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    logger.info(f"Experiment configuration saved to {config_path}")

def calculate_statistical_significance(
    metrics_a: List[float],
    metrics_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculate statistical significance between two sets of metrics."""
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(metrics_a) - 1) * np.var(metrics_a, ddof=1) + 
                         (len(metrics_b) - 1) * np.var(metrics_b, ddof=1)) / 
                        (len(metrics_a) + len(metrics_b) - 2))
    cohens_d = (np.mean(metrics_a) - np.mean(metrics_b)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }

def save_experiment_results(
    results: Dict[str, Any],
    experiment_name: str,
    save_dir: Path
) -> None:
    """Save comprehensive experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = save_dir / f"{experiment_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results.get('metrics', {}), f, indent=2)
    
    # Save model if available
    if 'model_state' in results:
        model_path = results_dir / "model.pth"
        torch.save(results['model_state'], model_path)
    
    # Save visualizations if available
    if 'visualizations' in results:
        viz_dir = results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        for name, fig in results['visualizations'].items():
            fig.savefig(viz_dir / f"{name}.png", dpi=300, bbox_inches='tight')
    
    # Save performance summary
    if 'performance_summary' in results:
        perf_path = results_dir / "performance_summary.json"
        with open(perf_path, 'w') as f:
            json.dump(results['performance_summary'], f, indent=2)
    
    logger.info(f"Experiment results saved to {results_dir}")

def validate_data_integrity(data_dir: Path) -> bool:
    """Validate the integrity of generated data."""
    try:
        # Check if directories exist
        train_images_dir = data_dir / "train" / "images"
        train_masks_dir = data_dir / "train" / "masks"
        val_images_dir = data_dir / "val" / "images"
        val_masks_dir = data_dir / "val" / "masks"
        
        for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
            if not dir_path.exists():
                logger.error(f"Directory does not exist: {dir_path}")
                return False
        
        # Check file counts
        train_images = len(list(train_images_dir.glob("*.png")))
        train_masks = len(list(train_masks_dir.glob("*.png")))
        val_images = len(list(val_images_dir.glob("*.png")))
        val_masks = len(list(val_masks_dir.glob("*.png")))
        
        if train_images != train_masks:
            logger.error(f"Mismatch in train data: {train_images} images vs {train_masks} masks")
            return False
        
        if val_images != val_masks:
            logger.error(f"Mismatch in validation data: {val_images} images vs {val_masks} masks")
            return False
        
        # Check file sizes
        for img_path in train_images_dir.glob("*.png"):
            if img_path.stat().st_size < 1000:  # Less than 1KB
                logger.warning(f"Suspiciously small image file: {img_path}")
        
        logger.info("Data integrity validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Data integrity validation failed: {e}")
        return False