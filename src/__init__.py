"""
AutoOpticalDiagnostics - Advanced Optical Diagnostics System
A comprehensive framework for synthetic data generation and AI-driven defect detection
using Laser Speckle Contrast Imaging (LSCI) and Optical Coherence Tomography (OCT).
"""

__version__ = "1.0.0"
__author__ = "AutoOpticalDiagnostics Team"
__email__ = "contact@autoopticaldiagnostics.com"

from .config import Config, ImagingMode, DefectType
from .utils import setup_logging, get_device_info, set_random_seed

__all__ = [
    "Config",
    "ImagingMode", 
    "DefectType",
    "setup_logging",
    "get_device_info",
    "set_random_seed"
]