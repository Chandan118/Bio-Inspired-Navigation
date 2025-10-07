"""
Configuration Management for Bio-Inspired Navigation Research

This module provides centralized configuration management for the project,
including paths, parameters, and settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for managing project settings and parameters.
    
    This class provides a centralized way to manage all configuration
    parameters for the bio-inspired navigation research project.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize configuration with optional base path.
        
        Args:
            base_path: Base path for the project (defaults to current directory)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Project structure
        self.output_dir = self.base_path / "output"
        self.plots_dir = self.base_path / "plots"
        self.data_dir = self.base_path / "data"
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Data generation parameters
        self.trajectory_params = {
            'n_points': 101,
            'distance_m': 100.0,
            'correction_interval': 20,
            'random_seed': 42
        }
        
        self.energy_params = {
            'n_trials': 100,
            'random_seed': 42
        }
        
        # Plotting parameters
        self.plot_params = {
            'dpi': 300,
            'figsize': (10, 8),
            'save_format': 'png',
            'style': 'seaborn-v0_8-whitegrid'
        }
        
        # File naming conventions
        self.file_names = {
            'trajectory_data': 'S1_localization_accuracy.csv',
            'energy_data': 'S1_energy_consumption.csv',
            'performance_data': 'S1_swarm_performance_summary.csv',
            'trajectory_plot': 'Figure13_Localization_Accuracy.png',
            'energy_plot': 'Figure15B_Energy_Consumption.png',
            'performance_plot': 'Performance_Comparison.png',
            'combined_plot': 'Combined_Results.png'
        }
        
        logger.info(f"Configuration initialized with base path: {self.base_path}")

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.output_dir, self.plots_dir, self.data_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def get_file_path(self, file_type: str) -> Path:
        """
        Get the full path for a specific file type.
        
        Args:
            file_type: Type of file (e.g., 'trajectory_data', 'energy_plot')
            
        Returns:
            Path: Full path to the file
        """
        if file_type not in self.file_names:
            raise ValueError(f"Unknown file type: {file_type}")
        
        filename = self.file_names[file_type]
        
        # Determine directory based on file type
        if 'plot' in file_type:
            directory = self.plots_dir
        elif 'data' in file_type:
            directory = self.output_dir
        else:
            directory = self.output_dir
        
        return directory / filename

    def get_trajectory_params(self) -> Dict[str, Any]:
        """Get trajectory generation parameters."""
        return self.trajectory_params.copy()

    def get_energy_params(self) -> Dict[str, Any]:
        """Get energy data generation parameters."""
        return self.energy_params.copy()

    def get_plot_params(self) -> Dict[str, Any]:
        """Get plotting parameters."""
        return self.plot_params.copy()

    def update_trajectory_params(self, **kwargs) -> None:
        """Update trajectory generation parameters."""
        self.trajectory_params.update(kwargs)
        logger.info(f"Updated trajectory parameters: {kwargs}")

    def update_energy_params(self, **kwargs) -> None:
        """Update energy data generation parameters."""
        self.energy_params.update(kwargs)
        logger.info(f"Updated energy parameters: {kwargs}")

    def update_plot_params(self, **kwargs) -> None:
        """Update plotting parameters."""
        self.plot_params.update(kwargs)
        logger.info(f"Updated plot parameters: {kwargs}")

    def get_all_paths(self) -> Dict[str, Path]:
        """Get all file paths as a dictionary."""
        paths = {}
        for file_type in self.file_names.keys():
            paths[file_type] = self.get_file_path(file_type)
        return paths

    def validate_config(self) -> Dict[str, bool]:
        """
        Validate the configuration for consistency and completeness.
        
        Returns:
            dict: Dictionary containing validation results
        """
        validation = {}
        
        # Check if directories exist
        validation['output_dir_exists'] = self.output_dir.exists()
        validation['plots_dir_exists'] = self.plots_dir.exists()
        validation['data_dir_exists'] = self.data_dir.exists()
        
        # Check if directories are writable
        validation['output_dir_writable'] = os.access(self.output_dir, os.W_OK)
        validation['plots_dir_writable'] = os.access(self.plots_dir, os.W_OK)
        validation['data_dir_writable'] = os.access(self.data_dir, os.W_OK)
        
        # Check parameter validity
        validation['trajectory_params_valid'] = (
            self.trajectory_params['n_points'] > 0 and
            self.trajectory_params['distance_m'] > 0 and
            self.trajectory_params['correction_interval'] > 0
        )
        
        validation['energy_params_valid'] = (
            self.energy_params['n_trials'] > 0
        )
        
        validation['plot_params_valid'] = (
            self.plot_params['dpi'] > 0 and
            len(self.plot_params['figsize']) == 2 and
            all(size > 0 for size in self.plot_params['figsize'])
        )
        
        return validation

    def print_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        
        print(f"\nBase Path: {self.base_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Plots Directory: {self.plots_dir}")
        print(f"Data Directory: {self.data_dir}")
        
        print(f"\nTrajectory Parameters:")
        for key, value in self.trajectory_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nEnergy Parameters:")
        for key, value in self.energy_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nPlot Parameters:")
        for key, value in self.plot_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nFile Names:")
        for key, value in self.file_names.items():
            print(f"  {key}: {value}")
        
        # Validation results
        validation = self.validate_config()
        print(f"\nValidation Results:")
        for key, value in validation.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        
        print("="*50 + "\n")


# Global configuration instance
config = Config() 