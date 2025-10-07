"""
Trajectory Data Generator for Bio-Inspired Navigation Research

This module generates sophisticated trajectory data that simulates different navigation
methods including ground truth, traditional SLAM with drift, and our proposed framework
with periodic error correction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrajectoryDataGenerator:
    """
    Generates sophisticated trajectory data matching Figure 13.
    
    This class simulates navigation trajectories for different methods:
    - Ground Truth: The ideal path
    - Traditional SLAM: Path with accumulating drift
    - Our Framework: Path with periodic error correction
    
    Attributes:
        n_points (int): Number of points in the trajectory
        distance_m (float): Total distance in meters
        correction_interval (int): Interval for error correction in our framework
        random_seed (int): Random seed for reproducibility
    """
    
    def __init__(self, 
                 n_points: int = 101, 
                 distance_m: float = 100.0, 
                 correction_interval: int = 20,
                 random_seed: int = 42):
        """
        Initialize the trajectory data generator.
        
        Args:
            n_points: Number of points in the trajectory (default: 101)
            distance_m: Total distance in meters (default: 100.0)
            correction_interval: Interval for error correction (default: 20)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.n_points = n_points
        self.distance_m = distance_m
        self.correction_interval = correction_interval
        self.random_seed = random_seed
        
        # Validate parameters
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if correction_interval < 1:
            raise ValueError("correction_interval must be at least 1")
        
        # Set random seed
        np.random.seed(self.random_seed)
        logger.info(f"TrajectoryDataGenerator initialized with {n_points} points, "
                   f"{distance_m}m distance, correction interval {correction_interval}")

    def _generate_ground_truth(self) -> pd.DataFrame:
        """Generate ground truth trajectory."""
        distances = np.linspace(0, self.distance_m, self.n_points)
        return pd.DataFrame({
            'x': distances,
            'y': np.zeros(self.n_points)
        })

    def _generate_traditional_slam(self, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Generate traditional SLAM trajectory with accumulating drift."""
        # Simulate drift that accumulates over time
        x_drift = np.cumsum(np.random.normal(0, 0.1, self.n_points))
        y_drift = np.cumsum(np.random.normal(0, 0.3, self.n_points))
        
        return pd.DataFrame({
            'x': ground_truth['x'] + x_drift,
            'y': ground_truth['y'] + y_drift
        })

    def _generate_our_framework(self, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Generate our framework trajectory with periodic error correction."""
        our_framework_x = np.zeros(self.n_points)
        our_framework_y = np.zeros(self.n_points)
        
        for i in range(1, self.n_points):
            if i % self.correction_interval == 0:
                # Error correction point - closer to ground truth
                our_framework_x[i] = ground_truth['x'].iloc[i] + np.random.normal(0, 0.05)
                our_framework_y[i] = ground_truth['y'].iloc[i] + np.random.normal(0, 0.2)
            else:
                # Regular navigation with small drift
                our_framework_x[i] = (our_framework_x[i-1] + 
                                    (ground_truth['x'].iloc[i] - ground_truth['x'].iloc[i-1]) + 
                                    np.random.normal(0, 0.08))
                our_framework_y[i] = our_framework_y[i-1] + np.random.normal(0, 0.1)
        
        return pd.DataFrame({
            'x': our_framework_x,
            'y': our_framework_y
        })

    def generate(self) -> pd.DataFrame:
        """
        Generate and return the complete trajectory dataframe.
        
        Returns:
            pd.DataFrame: DataFrame containing all trajectory data with columns:
                - distance_m: Distance along the path
                - ground_truth_x, ground_truth_y: Ground truth coordinates
                - our_framework_x, our_framework_y: Our framework coordinates
                - traditional_slam_x, traditional_slam_y: Traditional SLAM coordinates
        """
        logger.info("Generating trajectory data...")
        
        try:
            # Generate ground truth
            ground_truth = self._generate_ground_truth()
            
            # Generate traditional SLAM with drift
            traditional_slam = self._generate_traditional_slam(ground_truth)
            
            # Generate our framework with correction
            our_framework = self._generate_our_framework(ground_truth)
            
            # Combine all data
            result_df = pd.DataFrame({
                'distance_m': np.linspace(0, self.distance_m, self.n_points),
                'ground_truth_x': ground_truth['x'],
                'ground_truth_y': ground_truth['y'],
                'our_framework_x': our_framework['x'],
                'our_framework_y': our_framework['y'],
                'traditional_slam_x': traditional_slam['x'],
                'traditional_slam_y': traditional_slam['y']
            })
            
            logger.info(f"Successfully generated trajectory data with {len(result_df)} points")
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating trajectory data: {e}")
            raise

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate statistics for the generated trajectory data.
        
        Args:
            df: Trajectory dataframe
            
        Returns:
            dict: Dictionary containing various statistics
        """
        stats = {}
        
        # Calculate drift for traditional SLAM
        traditional_drift = np.sqrt(
            (df['traditional_slam_x'] - df['ground_truth_x'])**2 + 
            (df['traditional_slam_y'] - df['ground_truth_y'])**2
        )
        
        # Calculate error for our framework
        our_framework_error = np.sqrt(
            (df['our_framework_x'] - df['ground_truth_x'])**2 + 
            (df['our_framework_y'] - df['ground_truth_y'])**2
        )
        
        stats['traditional_slam_max_drift'] = traditional_drift.max()
        stats['traditional_slam_avg_drift'] = traditional_drift.mean()
        stats['our_framework_max_error'] = our_framework_error.max()
        stats['our_framework_avg_error'] = our_framework_error.mean()
        stats['improvement_ratio'] = stats['traditional_slam_avg_drift'] / stats['our_framework_avg_error']
        
        return stats 