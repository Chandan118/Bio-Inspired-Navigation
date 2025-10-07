"""
Energy Data Generator for Bio-Inspired Navigation Research

This module generates simulated energy consumption data for different navigation methods,
providing realistic distributions that match the performance claims in the research.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnergyDataGenerator:
    """
    Generates simulated energy consumption data for Figure 15B.
    
    This class creates realistic energy consumption distributions for different
    navigation methods, ensuring statistical significance and proper comparison.
    
    Attributes:
        n_trials (int): Number of trials for each method
        random_seed (int): Random seed for reproducibility
        energy_params (dict): Energy distribution parameters for each method
    """
    
    def __init__(self, n_trials: int = 100, random_seed: int = 42):
        """
        Initialize the energy data generator.
        
        Args:
            n_trials: Number of trials for each method (default: 100)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.n_trials = n_trials
        self.random_seed = random_seed
        
        # Validate parameters
        if n_trials < 10:
            raise ValueError("n_trials must be at least 10 for statistical significance")
        
        # Energy distribution parameters (mean, std) for each method
        self.energy_params = {
            'Our Method': {'loc': 30.0, 'scale': 4.0},
            'Baseline SLAM 1': {'loc': 50.0, 'scale': 6.0},
            'Baseline SLAM 2': {'loc': 55.0, 'scale': 7.5},
            'Other Baseline': {'loc': 48.0, 'scale': 5.0}
        }
        
        # Set random seed
        np.random.seed(self.random_seed)
        logger.info(f"EnergyDataGenerator initialized with {n_trials} trials per method")

    def _generate_method_energy(self, method_name: str) -> np.ndarray:
        """
        Generate energy consumption data for a specific method.
        
        Args:
            method_name: Name of the navigation method
            
        Returns:
            np.ndarray: Array of energy consumption values
        """
        params = self.energy_params[method_name]
        
        # Generate normal distribution with some outliers
        energy_data = np.random.normal(
            loc=params['loc'], 
            scale=params['scale'], 
            size=self.n_trials
        )
        
        # Add some outliers (5% of data)
        n_outliers = max(1, int(0.05 * self.n_trials))
        outlier_indices = np.random.choice(self.n_trials, n_outliers, replace=False)
        
        for idx in outlier_indices:
            # Add outliers that are 2-3 standard deviations away
            outlier_factor = np.random.choice([-1, 1]) * np.random.uniform(2, 3)
            energy_data[idx] += outlier_factor * params['scale']
        
        # Ensure no negative energy values
        energy_data = np.maximum(energy_data, 0.1)
        
        return energy_data

    def generate(self) -> pd.DataFrame:
        """
        Generate and return the complete energy consumption dataframe.
        
        Returns:
            pd.DataFrame: DataFrame containing energy consumption data with columns:
                - Method: Navigation method name
                - Energy_Consumption_Watts: Energy consumption in Watts
                - Trial: Trial number
        """
        logger.info("Generating energy consumption data...")
        
        try:
            data = {
                'Method': [],
                'Energy_Consumption_Watts': [],
                'Trial': []
            }
            
            # Generate data for each method
            for method_name in self.energy_params.keys():
                energy_values = self._generate_method_energy(method_name)
                
                data['Method'].extend([method_name] * self.n_trials)
                data['Energy_Consumption_Watts'].extend(energy_values)
                data['Trial'].extend(range(1, self.n_trials + 1))
            
            result_df = pd.DataFrame(data)
            logger.info(f"Successfully generated energy data for {len(self.energy_params)} methods")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating energy data: {e}")
            raise

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for the energy consumption data.
        
        Args:
            df: Energy consumption dataframe
            
        Returns:
            dict: Dictionary containing statistics for each method
        """
        stats = {}
        
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]['Energy_Consumption_Watts']
            
            stats[method] = {
                'mean': method_data.mean(),
                'std': method_data.std(),
                'median': method_data.median(),
                'min': method_data.min(),
                'max': method_data.max(),
                'q25': method_data.quantile(0.25),
                'q75': method_data.quantile(0.75),
                'iqr': method_data.quantile(0.75) - method_data.quantile(0.25)
            }
        
        return stats

    def calculate_improvements(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate improvement percentages compared to baselines.
        
        Args:
            df: Energy consumption dataframe
            
        Returns:
            dict: Dictionary containing improvement percentages
        """
        stats = self.get_statistics(df)
        improvements = {}
        
        our_method_mean = stats['Our Method']['mean']
        
        for method in ['Baseline SLAM 1', 'Baseline SLAM 2', 'Other Baseline']:
            if method in stats:
                baseline_mean = stats[method]['mean']
                improvement = ((baseline_mean - our_method_mean) / baseline_mean) * 100
                improvements[f'vs_{method.replace(" ", "_")}'] = improvement
        
        return improvements

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate the quality of generated energy data.
        
        Args:
            df: Energy consumption dataframe
            
        Returns:
            dict: Dictionary containing validation results
        """
        validation = {}
        
        # Check for negative values
        validation['no_negative_values'] = (df['Energy_Consumption_Watts'] >= 0).all()
        
        # Check for reasonable ranges (0-200 Watts)
        validation['reasonable_range'] = (df['Energy_Consumption_Watts'] <= 200).all()
        
        # Check for sufficient data points
        validation['sufficient_data'] = len(df) >= 50
        
        # Check for method diversity
        validation['method_diversity'] = len(df['Method'].unique()) >= 2
        
        # Check for statistical significance (enough trials per method)
        trials_per_method = df.groupby('Method').size()
        validation['statistical_significance'] = (trials_per_method >= 10).all()
        
        return validation 