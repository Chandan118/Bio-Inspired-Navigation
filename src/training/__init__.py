"""
Training package for AutoOpticalDiagnostics.
Contains training loops and utilities for model training.
"""

from .train import Trainer, TrainingConfig

__all__ = [
    "Trainer",
    "TrainingConfig"
]