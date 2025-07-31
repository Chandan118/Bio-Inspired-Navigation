"""
Utility functions for the AutoOpticalDiagnostics project.
Provides logging, device management, and common helper functions.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import psutil
import platform
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import cv2


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rich_console: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging with rich console output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        rich_console: Whether to use rich console formatting
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger("AutoOpticalDiagnostics")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if rich_console:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"auto_optical_diagnostics_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for the system.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "current_gpu": torch.cuda.current_device(),
            "gpu_memory_total": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
        })
    
    return info


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to set deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


def get_optimal_device() -> torch.device:
    """
    Get the optimal device for computation.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        # Check available memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 4 * 1024**3:  # 4GB
            return torch.device("cuda")
        else:
            print("GPU memory too low, using CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def create_progress_bar(description: str, total: int) -> Progress:
    """
    Create a rich progress bar.
    
    Args:
        description: Description for the progress bar
        total: Total number of iterations
        
    Returns:
        Progress bar object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=Console()
    )


def display_system_info() -> None:
    """Display comprehensive system information."""
    console = Console()
    
    # Get device info
    device_info = get_device_info()
    
    # Create table
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in device_info.items():
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)
        table.add_row(key, str(value))
    
    console.print(table)


def save_model_summary(model: nn.Module, file_path: str) -> None:
    """
    Save model summary to file.
    
    Args:
        model: PyTorch model
        file_path: Path to save the summary
    """
    param_counts = count_parameters(model)
    
    with open(file_path, 'w') as f:
        f.write("Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Total Parameters: {param_counts['total_parameters']:,}\n")
        f.write(f"Trainable Parameters: {param_counts['trainable_parameters']:,}\n")
        f.write(f"Non-trainable Parameters: {param_counts['non_trainable_parameters']:,}\n\n")
        
        f.write("Model Architecture:\n")
        f.write("-" * 20 + "\n")
        f.write(str(model))


def create_visualization_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a grid visualization of images.
    
    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        if len(img.shape) == 3 and img.shape[2] == 1:
            axes[row, col].imshow(img.squeeze(), cmap='gray')
        elif len(img.shape) == 3:
            axes[row, col].imshow(img)
        else:
            axes[row, col].imshow(img, cmap='gray')
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def apply_gaussian_noise(
    image: np.ndarray,
    std: float = 0.1,
    mean: float = 0.0
) -> np.ndarray:
    """
    Apply Gaussian noise to an image.
    
    Args:
        image: Input image
        std: Standard deviation of noise
        mean: Mean of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def apply_salt_pepper_noise(
    image: np.ndarray,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01
) -> np.ndarray:
    """
    Apply salt and pepper noise to an image.
    
    Args:
        image: Input image
        salt_prob: Probability of salt noise
        pepper_prob: Probability of pepper noise
        
    Returns:
        Noisy image
    """
    noisy_image = image.copy()
    
    # Salt noise
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 1
    
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    angle: float = 45
) -> np.ndarray:
    """
    Apply motion blur to an image.
    
    Args:
        image: Input image
        kernel_size: Size of motion blur kernel
        angle: Angle of motion in degrees
        
    Returns:
        Blurred image
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Create line in kernel
    for i in range(kernel_size):
        x = int(center + (i - center) * np.cos(angle_rad))
        y = int(center + (i - center) * np.sin(angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Apply convolution
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred[:, :, channel] = ndimage.convolve(
                image[:, :, channel], kernel, mode='reflect'
            )
    else:
        blurred = ndimage.convolve(image, kernel, mode='reflect')
    
    return blurred


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    
    if len(img1.shape) == 3:
        return ssim(img1, img2, multichannel=True)
    else:
        return ssim(img1, img2)


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create and display confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def create_roc_curve_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create and display ROC curve plot.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        class_names: Names of classes
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    if len(y_true.shape) == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class classification
        for i in range(y_true.shape[1]):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def create_precision_recall_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create and display Precision-Recall curve plot.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        class_names: Names of classes
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    if len(y_true.shape) == 1:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    else:
        # Multi-class classification
        for i in range(y_true.shape[1]):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
            avg_precision = average_precision_score(y_true[:, i], y_scores[:, i])
            
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def create_performance_report(
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Model Performance Report"
) -> None:
    """
    Create a comprehensive performance report.
    
    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save the report
        title: Report title
    """
    console = Console()
    
    # Create table
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))
    
    # Display in console
    console.print(table)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
    
    print(f"Performance report saved to {save_path}")