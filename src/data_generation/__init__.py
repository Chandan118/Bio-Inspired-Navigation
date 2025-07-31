"""
Data generation package for AutoOpticalDiagnostics.
Contains modules for generating synthetic optical imaging data.
"""

from .oct_simulator import OCTSimulator
from .lsci_simulator import LSCISimulator
from .noise_models import NoiseModel
from .main_generator import DataGenerator

__all__ = [
    "OCTSimulator",
    "LSCISimulator", 
    "NoiseModel",
    "DataGenerator"
]