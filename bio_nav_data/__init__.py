"""
__init__.py

Author      : Chandan Sheikder
Email       : chandan@bit.edu.cn
Phone       : +8618222390506
Affiliation : Beijing Institute of Technology (BIT)
Date        : 2026-03-23

Description:
    Bio-Inspired Navigation Data Generation Package
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
