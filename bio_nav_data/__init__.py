"""
Bio-Inspired Navigation Data Generation Package

This package provides tools for generating and visualizing bio-inspired navigation data
for research and analysis purposes.

Author: Research Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Research Team"

from .data_generators.energy import EnergyDataGenerator
from .data_generators.performance import create_performance_summary
from .data_generators.trajectory import TrajectoryDataGenerator
from .pipeline import BioNavDataGenerator
from .utils.config import Config
from .visualizers.plots import plot_energy_consumption, plot_trajectory

__all__ = [
    "BioNavDataGenerator",
    "TrajectoryDataGenerator",
    "EnergyDataGenerator",
    "create_performance_summary",
    "plot_trajectory",
    "plot_energy_consumption",
    "Config",
]
