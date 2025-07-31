"""
Training module for AutoOpticalDiagnostics.
Implements comprehensive training loop with advanced features.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import wandb
from datetime import datetime

from ..model import UNet, create_unet_model, create_loss_function
from ..model.dataset import OpticalDiagnosticsDataset, create_train_transforms, create_val_transforms
from ..utils import (
    setup_device, set_random_seed, save_model_checkpoint, load_model_checkpoint,
    calculate_metrics, visualize_predictions, PerformanceMonitor, print_system_info
)
from ..config import get_config

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping mechanism to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

class LRScheduler:
    """Learning rate scheduler with various strategies."""
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = "plateau", **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == "warmup_cosine":
            self.scheduler = self._create_warmup_cosine_scheduler(kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _create_warmup_cosine_scheduler(self, kwargs):
        """Create warmup cosine scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR
        
        return OneCycleLR(
            self.optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            epochs=kwargs.get('epochs', 100),
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
            pct_start=kwargs.get('pct_start', 0.1)
        )
    
    def step(self, val_loss: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == "plateau":
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()

class Trainer:
    """
    Comprehensive trainer for optical diagnostics models.
    
    Features:
    - Advanced training loop with validation
    - Early stopping and learning rate scheduling
    - Comprehensive logging and monitoring
    - Model checkpointing and restoration
    - Performance monitoring
    - Multi-GPU support
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = setup_device(self.config.training.device)
        
        # Set random seed
        set_random_seed(self.config.data_generation.random_seed)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Logging
        self.writer = None
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_model(self) -> None:
        """Setup model, optimizer, and loss function."""
        # Create model
        self.model = create_unet_model(self.config)
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Create loss function
        self.criterion = create_loss_function(self.config)
        
        # Create scheduler
        self.scheduler = LRScheduler(
            self.optimizer,
            scheduler_type="plateau",
            factor=self.config.model.lr_scheduler_factor,
            patience=self.config.model.lr_scheduler_patience
        )
        
        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.model.early_stopping_patience
        )
        
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Model size: {self.model.get_model_size_mb():.2f} MB")
    
    def setup_data(self, data_dir: Path) -> None:
        """Setup data loaders."""
        # Create transforms
        train_transforms = create_train_transforms(self.config)
        val_transforms = create_val_transforms(self.config)
        
        # Create datasets
        train_dataset = OpticalDiagnosticsDataset(
            data_dir=data_dir,
            split="train",
            transform=train_transforms,
            modalities=self.config.data_generation.materials,
            max_samples=self.config.data_generation.num_train_samples
        )
        
        val_dataset = OpticalDiagnosticsDataset(
            data_dir=data_dir,
            split="val",
            transform=val_transforms,
            modalities=self.config.data_generation.materials,
            max_samples=self.config.data_generation.num_val_samples
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def setup_logging(self, log_dir: Path) -> None:
        """Setup logging and monitoring."""
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.training.tensorboard_logging:
            self.writer = SummaryWriter(log_dir / "tensorboard")
        
        # Weights & Biases
        if self.config.training.wandb_logging:
            wandb.init(
                project=self.config.training.wandb_project,
                config=self.config.__dict__,
                name=f"optical_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Performance monitoring
        self.performance_monitor.start()
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        epoch_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(self.criterion, nn.Module):
                    loss = self.criterion(outputs, masks)
                else:
                    loss_dict = self.criterion(outputs, masks)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs)
                    batch_metrics = calculate_metrics(predictions, masks)
                
                # Update running totals
                total_loss += loss.item()
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{batch_metrics['dice']:.4f}"
                })
                
                # Log to TensorBoard
                if self.writer and batch_idx % self.config.training.log_interval == 0:
                    step = self.current_epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                    self.writer.add_scalar('Train/BatchDice', batch_metrics['dice'], step)
                
                # Performance monitoring
                self.performance_monitor.record()
        
        # Average metrics
        avg_loss = total_loss / num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        epoch_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move data to device
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    if isinstance(self.criterion, nn.Module):
                        loss = self.criterion(outputs, masks)
                    else:
                        loss_dict = self.criterion(outputs, masks)
                        loss = loss_dict['total_loss']
                    
                    # Calculate metrics
                    predictions = torch.sigmoid(outputs)
                    batch_metrics = calculate_metrics(predictions, masks)
                    
                    # Update running totals
                    total_loss += loss.item()
                    for key in epoch_metrics:
                        epoch_metrics[key] += batch_metrics[key]
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'dice': f"{batch_metrics['dice']:.4f}"
                    })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return avg_loss, epoch_metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save regular checkpoint
        if self.config.training.save_last_checkpoint:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
            save_model_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.best_val_loss,
                self.best_val_metrics,
                str(checkpoint_path)
            )
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            save_model_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                self.best_val_loss,
                self.best_val_metrics,
                str(best_path),
                is_best=True
            )
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.current_epoch, self.best_val_loss, self.best_val_metrics = load_model_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path
        )
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def log_epoch(self, train_loss: float, val_loss: float, 
                  train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log epoch results."""
        # Update history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to console
        logger.info(f"Epoch {self.current_epoch + 1}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"  Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
            
            for metric in train_metrics:
                self.writer.add_scalar(f'Train/{metric.capitalize()}', train_metrics[metric], self.current_epoch)
                self.writer.add_scalar(f'Validation/{metric.capitalize()}', val_metrics[metric], self.current_epoch)
        
        # Log to Weights & Biases
        if self.config.training.wandb_logging:
            wandb.log({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
    
    def train(self, data_dir: Path, output_dir: Path, num_epochs: int = None) -> Dict[str, Any]:
        """Main training loop."""
        # Setup
        self.setup_model()
        self.setup_data(data_dir)
        self.setup_logging(output_dir)
        
        # Print system info
        print_system_info()
        
        num_epochs = num_epochs or self.config.model.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Output directory: {output_dir}")
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch()
            
            # Log results
            self.log_epoch(train_loss, val_loss, train_metrics, val_metrics)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics.copy()
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                break
        
        # Final performance summary
        performance_summary = self.performance_monitor.get_summary()
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation dice: {self.best_val_metrics['dice']:.4f}")
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Close logging
        if self.writer:
            self.writer.close()
        if self.config.training.wandb_logging:
            wandb.finish()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'performance_summary': performance_summary
        }

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train optical diagnostics model")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer()
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    results = trainer.train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        num_epochs=args.epochs
    )
    
    print("Training completed successfully!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()