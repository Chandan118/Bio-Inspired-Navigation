#!/usr/bin/env python3
"""
AutoOpticalDiagnostics - Main Pipeline Script

Advanced AI-driven optical diagnostics for industrial quality control.
This script orchestrates the complete pipeline from data generation to model evaluation.

Author: AutoOpticalDiagnostics Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, get_development_config, get_production_config
from src.utils import (
    setup_device, set_random_seed, create_directory_structure, 
    print_system_info, PerformanceMonitor, console
)
from src.data_generation import DataGenerator
from src.training import Trainer
from src.evaluation import ModelEvaluator

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

class AutoOpticalDiagnosticsPipeline:
    """
    Main pipeline for AutoOpticalDiagnostics.
    
    Orchestrates the complete workflow:
    1. Data Generation (LSCI/OCT simulation)
    2. Model Training (U-Net with attention)
    3. Model Evaluation (Comprehensive metrics)
    4. Results Analysis (Visualizations and reports)
    """
    
    def __init__(self, config=None, mode="production"):
        self.config = config or get_config()
        self.mode = mode
        
        # Use appropriate config based on mode
        if mode == "development":
            self.config = get_development_config()
        elif mode == "production":
            self.config = get_production_config()
        
        # Setup device and random seed
        self.device = setup_device(self.config.training.device)
        set_random_seed(self.config.data_generation.random_seed)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Pipeline state
        self.pipeline_results = {
            'data_generation': {},
            'training': {},
            'evaluation': {},
            'performance_summary': {},
            'timestamp': datetime.now().isoformat(),
            'mode': mode
        }
        
        logger.info(f"AutoOpticalDiagnostics Pipeline initialized in {mode} mode")
        logger.info(f"Using device: {self.device}")
    
    def run_data_generation(self, output_dir: Path, **kwargs) -> Dict:
        """Run synthetic data generation."""
        logger.info("Starting data generation phase...")
        
        # Create data generator
        generator = DataGenerator(self.config)
        
        # Generate dataset
        generation_stats = generator.generate_complete_dataset(
            output_dir=output_dir,
            **kwargs
        )
        
        self.pipeline_results['data_generation'] = generation_stats
        
        logger.info("Data generation completed successfully!")
        return generation_stats
    
    def run_training(self, data_dir: Path, output_dir: Path, **kwargs) -> Dict:
        """Run model training."""
        logger.info("Starting model training phase...")
        
        # Create trainer
        trainer = Trainer(self.config)
        
        # Train model
        training_results = trainer.train(
            data_dir=data_dir,
            output_dir=output_dir,
            **kwargs
        )
        
        self.pipeline_results['training'] = training_results
        
        logger.info("Model training completed successfully!")
        return training_results
    
    def run_evaluation(self, model_path: str, data_dir: Path, output_dir: Path, **kwargs) -> Dict:
        """Run model evaluation."""
        logger.info("Starting model evaluation phase...")
        
        # Create evaluator
        evaluator = ModelEvaluator(self.config)
        
        # Load model
        evaluator.load_model(model_path)
        
        # Setup test data
        evaluator.setup_test_data(data_dir)
        
        # Evaluate model
        evaluation_results = evaluator.evaluate_model(**kwargs)
        
        # Optimize threshold
        threshold_results = evaluator.optimize_threshold(
            evaluation_results['predictions'],
            evaluation_results['targets']
        )
        evaluation_results['threshold_optimization'] = threshold_results
        
        # Generate visualizations and reports
        evaluator.generate_visualizations(evaluation_results, output_dir)
        evaluator.generate_report(evaluation_results, output_dir)
        
        self.pipeline_results['evaluation'] = evaluation_results
        
        logger.info("Model evaluation completed successfully!")
        return evaluation_results
    
    def run_complete_pipeline(self, base_dir: Path, **kwargs) -> Dict:
        """Run the complete pipeline from start to finish."""
        logger.info("Starting complete AutoOpticalDiagnostics pipeline...")
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        # Create directory structure
        create_directory_structure(base_dir)
        
        # Print system information
        print_system_info()
        
        try:
            # Phase 1: Data Generation
            data_dir = base_dir / "data"
            generation_stats = self.run_data_generation(data_dir, **kwargs)
            
            # Phase 2: Model Training
            training_dir = base_dir / "training_outputs"
            training_results = self.run_training(data_dir, training_dir, **kwargs)
            
            # Phase 3: Model Evaluation
            model_path = base_dir / "checkpoints" / "best_model.pth"
            evaluation_dir = base_dir / "evaluation_results"
            evaluation_results = self.run_evaluation(
                str(model_path), data_dir, evaluation_dir, **kwargs
            )
            
            # Record performance summary
            self.performance_monitor.record()
            self.pipeline_results['performance_summary'] = self.performance_monitor.get_summary()
            
            # Save pipeline results
            results_path = base_dir / "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            
            logger.info("Complete pipeline executed successfully!")
            self._print_summary()
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        console.print("\n" + "="*80)
        console.print("[bold green]AutoOpticalDiagnostics Pipeline Summary[/bold green]")
        console.print("="*80)
        
        # Data Generation Summary
        if 'data_generation' in self.pipeline_results:
            gen_stats = self.pipeline_results['data_generation']
            console.print(f"\n[bold cyan]Data Generation:[/bold cyan]")
            console.print(f"  Training samples: {gen_stats.get('train_samples', 'N/A')}")
            console.print(f"  Validation samples: {gen_stats.get('val_samples', 'N/A')}")
            console.print(f"  Imaging modalities: {gen_stats.get('imaging_modalities', 'N/A')}")
        
        # Training Summary
        if 'training' in self.pipeline_results:
            train_results = self.pipeline_results['training']
            console.print(f"\n[bold cyan]Model Training:[/bold cyan]")
            console.print(f"  Best validation loss: {train_results.get('best_val_loss', 'N/A'):.4f}")
            console.print(f"  Best validation dice: {train_results.get('best_val_metrics', {}).get('dice', 'N/A'):.4f}")
        
        # Evaluation Summary
        if 'evaluation' in self.pipeline_results:
            eval_results = self.pipeline_results['evaluation']
            overall_metrics = eval_results.get('overall_metrics', {})
            console.print(f"\n[bold cyan]Model Evaluation:[/bold cyan]")
            console.print(f"  F1 Score: {overall_metrics.get('f1_score', 'N/A'):.4f}")
            console.print(f"  Dice Score: {overall_metrics.get('dice_score', 'N/A'):.4f}")
            console.print(f"  IoU: {overall_metrics.get('iou', 'N/A'):.4f}")
            console.print(f"  Precision: {overall_metrics.get('precision', 'N/A'):.4f}")
            console.print(f"  Recall: {overall_metrics.get('recall', 'N/A'):.4f}")
        
        # Performance Summary
        if 'performance_summary' in self.pipeline_results:
            perf_summary = self.pipeline_results['performance_summary']
            console.print(f"\n[bold cyan]Performance Summary:[/bold cyan]")
            console.print(f"  Total execution time: {perf_summary.get('total_time_seconds', 'N/A'):.2f} seconds")
            console.print(f"  Peak memory usage: {perf_summary.get('memory', {}).get('peak_used_gb', 'N/A'):.2f} GB")
        
        console.print("\n" + "="*80)

def main():
    """Main function for the AutoOpticalDiagnostics pipeline."""
    parser = argparse.ArgumentParser(
        description="AutoOpticalDiagnostics - Advanced AI-driven optical diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline in development mode
  python main.py --mode development --output_dir ./outputs

  # Run only data generation
  python main.py --phase data_generation --output_dir ./data

  # Run only training with custom parameters
  python main.py --phase training --data_dir ./data --output_dir ./training

  # Run only evaluation
  python main.py --phase evaluation --model_path ./checkpoints/best_model.pth --data_dir ./data
        """
    )
    
    # Main arguments
    parser.add_argument("--mode", choices=["development", "production"], default="production",
                       help="Pipeline mode (development/production)")
    parser.add_argument("--phase", choices=["all", "data_generation", "training", "evaluation"],
                       default="all", help="Pipeline phase to run")
    parser.add_argument("--output_dir", type=str, default="./auto_optical_diagnostics_outputs",
                       help="Output directory for all results")
    
    # Data generation arguments
    parser.add_argument("--train_samples", type=int, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, help="Number of validation samples")
    parser.add_argument("--modalities", nargs="+", help="Imaging modalities")
    parser.add_argument("--environments", nargs="+", help="Environments")
    parser.add_argument("--materials", nargs="+", help="Materials")
    parser.add_argument("--defect_types", nargs="+", help="Defect types")
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, help="Data directory for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    
    # Evaluation arguments
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutoOpticalDiagnosticsPipeline(mode=args.mode)
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.phase == "all":
            # Run complete pipeline
            pipeline.run_complete_pipeline(
                output_dir,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                imaging_modalities=args.modalities,
                environments=args.environments,
                materials=args.materials,
                defect_types=args.defect_types,
                num_epochs=args.epochs
            )
        
        elif args.phase == "data_generation":
            # Run only data generation
            data_dir = output_dir / "data"
            pipeline.run_data_generation(
                data_dir,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                imaging_modalities=args.modalities,
                environments=args.environments,
                materials=args.materials,
                defect_types=args.defect_types
            )
        
        elif args.phase == "training":
            # Run only training
            if not args.data_dir:
                raise ValueError("--data_dir is required for training phase")
            
            training_dir = output_dir / "training_outputs"
            pipeline.run_training(
                Path(args.data_dir),
                training_dir,
                num_epochs=args.epochs
            )
        
        elif args.phase == "evaluation":
            # Run only evaluation
            if not args.model_path or not args.data_dir:
                raise ValueError("--model_path and --data_dir are required for evaluation phase")
            
            evaluation_dir = output_dir / "evaluation_results"
            pipeline.run_evaluation(
                args.model_path,
                Path(args.data_dir),
                evaluation_dir,
                threshold=args.threshold
            )
        
        console.print(f"\n[bold green]Pipeline execution completed successfully![/bold green]")
        console.print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"\n[bold red]Pipeline execution failed: {e}[/bold red]")
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()