"""
Training module for AutoOpticalDiagnostics.
Contains training loop, validation, and model optimization components.
"""

from .train import Trainer, TrainingConfig

__all__ = [
    'Trainer',
    'TrainingConfig'
]