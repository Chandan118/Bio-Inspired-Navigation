"""
Data generation module for AutoOpticalDiagnostics.
Contains synthetic data generators for LSCI and OCT imaging modalities.
"""

from .oct_simulator import OCTSimulator
from .lsci_simulator import LSCISimulator
from .noise_models import NoiseModel
from .main_generator import DataGenerator

__all__ = [
    'OCTSimulator',
    'LSCISimulator', 
    'NoiseModel',
    'DataGenerator'
]