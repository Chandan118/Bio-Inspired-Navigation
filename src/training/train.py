"""
Advanced training module for optical defect detection models.
Includes mixed precision training, gradient clipping, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ..model.unet_model import create_model, count_parameters
from ..model.loss import create_loss_function
from ..model.dataset import create_data_loaders
from ..utils import setup_logging, get_device_info, format_time, create_performance_report


class Trainer:
    """
    Advanced trainer for optical defect detection models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rate': []
        }
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup logging and checkpoints
        self.setup_logging_and_checkpoints()
        
        # Display training info
        self.display_training_info()
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config.get('model', {})
        model = create_model(
            model_type=model_config.get('model_type', 'unet'),
            n_channels=model_config.get('input_channels', 3),
            n_classes=model_config.get('output_channels', 1),
            initial_features=model_config.get('initial_features', 64),
            depth=model_config.get('depth', 5),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            batch_norm=model_config.get('batch_norm', True),
            bilinear=model_config.get('bilinear', True)
        )
        
        self.logger.info(f"Created {model_config.get('model_type', 'unet')} model")
        self.logger.info(f"Model parameters: {count_parameters(model)}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config.get('loss', {})
        loss_fn = create_loss_function(
            loss_type=loss_config.get('loss_type', 'dice_bce'),
            dice_weight=loss_config.get('dice_weight', 0.5),
            bce_weight=loss_config.get('bce_weight', 0.5),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )
        
        self.logger.info(f"Created {loss_config.get('loss_type', 'dice_bce')} loss function")
        return loss_fn
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('learning_rate', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('learning_rate', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-5),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('learning_rate', 1e-3),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        self.logger.info(f"Created {optimizer_type} optimizer")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_lr_on_plateau')
        
        if scheduler_type == 'reduce_lr_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        if scheduler:
            self.logger.info(f"Created {scheduler_type} scheduler")
        
        return scheduler
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders based on configuration."""
        data_config = self.config.get('data', {})
        
        train_loader, val_loader = create_data_loaders(
            train_images_dir=data_config.get('train_images_dir', 'data/synthetic/train/images'),
            train_masks_dir=data_config.get('train_masks_dir', 'data/synthetic/train/masks'),
            val_images_dir=data_config.get('val_images_dir', 'data/synthetic/val/images'),
            val_masks_dir=data_config.get('val_masks_dir', 'data/synthetic/val/masks'),
            train_metadata_file=data_config.get('train_metadata_file'),
            val_metadata_file=data_config.get('val_metadata_file'),
            batch_size=data_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            image_size=data_config.get('image_size', (512, 512)),
            augmentation_probability=data_config.get('augmentation_probability', 0.8),
            pin_memory=data_config.get('pin_memory', True)
        )
        
        self.logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
    def setup_logging_and_checkpoints(self):
        """Setup logging and checkpoint directories."""
        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard logging
        if self.config.get('tensorboard_logging', True):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.logs_dir / 'tensorboard')
        else:
            self.writer = None
    
    def display_training_info(self):
        """Display comprehensive training information."""
        # Create info table
        table = Table(title="Training Configuration")
        table.add_column("Component", style="cyan")
        table.add_column("Configuration", style="magenta")
        
        # Model info
        param_counts = count_parameters(self.model)
        table.add_row("Model", f"{self.config.get('model', {}).get('model_type', 'unet')} "
                      f"({param_counts['trainable_parameters']:,} parameters)")
        
        # Device info
        table.add_row("Device", str(self.device))
        
        # Training info
        table.add_row("Epochs", str(self.config.get('num_epochs', 100)))
        table.add_row("Batch Size", str(self.config.get('data', {}).get('batch_size', 8)))
        table.add_row("Learning Rate", str(self.config.get('optimizer', {}).get('learning_rate', 1e-4)))
        
        # Loss info
        table.add_row("Loss Function", self.config.get('loss', {}).get('loss_type', 'dice_bce'))
        
        # Data info
        table.add_row("Train Samples", str(len(self.train_loader.dataset)))
        table.add_row("Val Samples", str(len(self.val_loader.dataset)))
        
        self.console.print(table)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(self.train_loader)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("Training...", total=num_batches)
            
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        metric = self.calculate_metric(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    metric = self.calculate_metric(outputs, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.get('gradient_clipping', True):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.get('gradient_clipping', True):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_metric += metric
                
                # Update progress
                progress.update(task, advance=1)
                
                # Log batch metrics
                if batch_idx % self.config.get('log_interval', 10) == 0:
                    self.logger.info(
                        f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                        f"Loss: {loss.item():.4f}, Metric: {metric:.4f}"
                    )
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return {'loss': avg_loss, 'metric': avg_metric}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                task = progress.add_task("Validating...", total=num_batches)
                
                for batch_idx, batch in enumerate(self.val_loader):
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    # Forward pass
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(images)
                            loss = self.criterion(outputs, masks)
                            metric = self.calculate_metric(outputs, masks)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        metric = self.calculate_metric(outputs, masks)
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_metric += metric
                    
                    # Update progress
                    progress.update(task, advance=1)
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return {'loss': avg_loss, 'metric': avg_metric}
    
    def calculate_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate metric (Dice score)."""
        # Convert to probabilities
        probs = torch.sigmoid(outputs)
        
        # Threshold
        preds = (probs > 0.5).float()
        
        # Calculate Dice score
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
        
        return dice_score.item()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint['best_val_metric']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Metric plot
        axes[0, 1].plot(self.training_history['train_metric'], label='Train Metric')
        axes[0, 1].plot(self.training_history['val_metric'], label='Val Metric')
        axes[0, 1].set_title('Training and Validation Metric')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss vs Metric
        axes[1, 1].scatter(self.training_history['val_loss'], self.training_history['val_metric'])
        axes[1, 1].set_title('Validation Loss vs Metric')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Metric')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        num_epochs = self.config.get('num_epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_metric'].append(train_metrics['metric'])
            self.training_history['val_metric'].append(val_metrics['metric'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Metric: {train_metrics['metric']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Metric: {val_metrics['metric']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Metric/Train', train_metrics['metric'], epoch)
                self.writer.add_scalar('Metric/Val', val_metrics['metric'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model
            is_best = False
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_metric = val_metrics['metric']
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        # Final cleanup
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        
        # Save final model
        self.save_checkpoint()
        
        # Plot training history
        self.plot_training_history()
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        # Create final report
        self.create_training_report()
    
    def create_training_report(self):
        """Create comprehensive training report."""
        report = {
            'training_config': self.config,
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_metric': self.best_val_metric,
                'final_train_loss': self.training_history['train_loss'][-1],
                'final_train_metric': self.training_history['train_metric'][-1],
                'final_val_loss': self.training_history['val_loss'][-1],
                'final_val_metric': self.training_history['val_metric'][-1]
            },
            'training_history': self.training_history,
            'model_info': count_parameters(self.model),
            'device_info': get_device_info()
        }
        
        # Save report
        report_path = self.output_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training report saved to {report_path}")


def create_training_config(
    model_type: str = "unet",
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    loss_type: str = "dice_bce"
) -> Dict[str, Any]:
    """
    Create training configuration.
    
    Args:
        model_type: Type of model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        loss_type: Type of loss function
        
    Returns:
        Training configuration dictionary
    """
    return {
        'model': {
            'model_type': model_type,
            'input_channels': 3,
            'output_channels': 1,
            'initial_features': 64,
            'depth': 5,
            'dropout_rate': 0.2,
            'batch_norm': True,
            'bilinear': True
        },
        'loss': {
            'loss_type': loss_type,
            'dice_weight': 0.5,
            'bce_weight': 0.5,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        },
        'optimizer': {
            'type': 'adam',
            'learning_rate': learning_rate,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'reduce_lr_on_plateau',
            'factor': 0.5,
            'patience': 10
        },
        'data': {
            'batch_size': batch_size,
            'num_workers': 4,
            'image_size': (512, 512),
            'augmentation_probability': 0.8,
            'pin_memory': True
        },
        'training': {
            'num_epochs': num_epochs,
            'early_stopping_patience': 15,
            'gradient_clipping': True,
            'max_grad_norm': 1.0,
            'use_amp': True,
            'log_interval': 10,
            'save_interval': 5,
            'tensorboard_logging': True
        },
        'output_dir': 'outputs',
        'device': 'auto'
    }