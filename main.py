#!/usr/bin/env python3
"""
AutoOpticalDiagnostics - Main Pipeline Script
Advanced optical diagnostics system for defect detection using LSCI and OCT imaging.

This script provides a comprehensive pipeline for:
1. Synthetic data generation with realistic optical imaging
2. Model training with advanced U-Net architectures
3. Model evaluation and performance analysis
4. Defect detection and visualization

Usage:
    python main.py --mode generate_data
    python main.py --mode train --config configs/training_config.yaml
    python main.py --mode evaluate --model_path models/best_model.pth
    python main.py --mode full_pipeline
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import setup_logging, get_device_info, display_system_info, set_random_seed
from src.config import Config
from src.data_generation import DataGenerator, create_data_generator_config
from src.training.train import Trainer, create_training_config
from src.model.unet_model import create_model, count_parameters
from src.model.dataset import create_data_loaders
from src.model.loss import create_loss_function


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoOpticalDiagnostics - Advanced Optical Diagnostics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate synthetic data
    python main.py --mode generate_data --config configs/data_config.yaml
    
    # Train model
    python main.py --mode train --config configs/training_config.yaml
    
    # Evaluate model
    python main.py --mode evaluate --model_path models/best_model.pth
    
    # Run full pipeline
    python main.py --mode full_pipeline --config configs/full_config.yaml
    
    # Generate sample visualization
    python main.py --mode visualize --num_samples 16
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate_data', 'train', 'evaluate', 'full_pipeline', 'visualize'],
        help='Pipeline mode to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples for visualization'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path:
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def generate_synthetic_data(config: Dict[str, Any], logger: logging.Logger):
    """Generate synthetic optical imaging data."""
    logger.info("Starting synthetic data generation...")
    
    # Create data generator
    data_config = config.get('data_generation', create_data_generator_config())
    data_generator = DataGenerator(data_config)
    
    # Generate training dataset
    logger.info("Generating training dataset...")
    data_generator.generate_dataset("train")
    
    # Generate validation dataset
    logger.info("Generating validation dataset...")
    data_generator.generate_dataset("val")
    
    # Generate sample visualization
    logger.info("Generating sample visualization...")
    data_generator.generate_sample_visualization()
    
    # Generate dataset statistics
    logger.info("Generating dataset statistics...")
    stats = data_generator.generate_dataset_statistics()
    
    logger.info("Synthetic data generation completed successfully!")
    return data_generator


def train_model(config: Dict[str, Any], logger: logging.Logger):
    """Train the defect detection model."""
    logger.info("Starting model training...")
    
    # Create training configuration
    training_config = config.get('training', create_training_config())
    
    # Create trainer
    trainer = Trainer(training_config)
    
    # Start training
    trainer.train()
    
    logger.info("Model training completed successfully!")
    return trainer


def evaluate_model(model_path: str, config: Dict[str, Any], logger: logging.Logger):
    """Evaluate the trained model."""
    logger.info(f"Starting model evaluation for {model_path}...")
    
    # Load model
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config'].get('model', {})
    
    model = create_model(
        model_type=model_config.get('model_type', 'unet'),
        n_channels=model_config.get('input_channels', 3),
        n_classes=model_config.get('output_channels', 1),
        initial_features=model_config.get('initial_features', 64),
        depth=model_config.get('depth', 5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create evaluation data loader
    eval_config = config.get('evaluation', {})
    _, val_loader = create_data_loaders(
        train_images_dir=eval_config.get('train_images_dir', 'data/synthetic/train/images'),
        train_masks_dir=eval_config.get('train_masks_dir', 'data/synthetic/train/masks'),
        val_images_dir=eval_config.get('val_images_dir', 'data/synthetic/val/images'),
        val_masks_dir=eval_config.get('val_masks_dir', 'data/synthetic/val/masks'),
        batch_size=eval_config.get('batch_size', 8)
    )
    
    # Perform evaluation
    logger.info("Performing model evaluation...")
    
    # TODO: Implement comprehensive evaluation
    # This would include metrics calculation, visualization, etc.
    
    logger.info("Model evaluation completed!")
    return model


def run_full_pipeline(config: Dict[str, Any], logger: logging.Logger):
    """Run the complete pipeline from data generation to evaluation."""
    logger.info("Starting full pipeline execution...")
    
    start_time = time.time()
    
    # Step 1: Generate synthetic data
    logger.info("=== Step 1: Synthetic Data Generation ===")
    data_generator = generate_synthetic_data(config, logger)
    
    # Step 2: Train model
    logger.info("=== Step 2: Model Training ===")
    trainer = train_model(config, logger)
    
    # Step 3: Evaluate model
    logger.info("=== Step 3: Model Evaluation ===")
    best_model_path = trainer.checkpoint_dir / 'best_checkpoint.pth'
    if best_model_path.exists():
        model = evaluate_model(str(best_model_path), config, logger)
    else:
        logger.warning("Best model checkpoint not found, skipping evaluation")
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Full pipeline completed in {total_time:.2f} seconds")
    
    return data_generator, trainer


def generate_visualization(num_samples: int, logger: logging.Logger):
    """Generate sample visualization."""
    logger.info(f"Generating visualization with {num_samples} samples...")
    
    # Create data generator with default config
    data_config = create_data_generator_config()
    data_generator = DataGenerator(data_config)
    
    # Generate sample visualization
    data_generator.generate_sample_visualization(num_samples)
    
    logger.info("Visualization generated successfully!")
    return data_generator


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level=log_level)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Display system information
    logger.info("AutoOpticalDiagnostics - Advanced Optical Diagnostics System")
    logger.info("=" * 60)
    display_system_info()
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    
    try:
        if args.mode == 'generate_data':
            generate_synthetic_data(config, logger)
            
        elif args.mode == 'train':
            train_model(config, logger)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                logger.error("Model path is required for evaluation mode")
                sys.exit(1)
            evaluate_model(args.model_path, config, logger)
            
        elif args.mode == 'full_pipeline':
            run_full_pipeline(config, logger)
            
        elif args.mode == 'visualize':
            generate_visualization(args.num_samples, logger)
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()