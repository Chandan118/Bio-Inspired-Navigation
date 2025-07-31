"""
Evaluation package for AutoOpticalDiagnostics.
Contains evaluation metrics and visualization tools for model assessment.
"""

from .evaluate import Evaluator, EvaluationMetrics, create_evaluation_config

__all__ = [
    "Evaluator",
    "EvaluationMetrics",
    "create_evaluation_config"
]