"""
Configuration module for AutoOpticalDiagnostics project.
Contains all hyperparameters, paths, and settings for the optical diagnostics system.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
from dataclasses import dataclass, field
from enum import Enum


class ImagingMode(Enum):
    """Enumeration for different imaging modalities."""
    LSCI = "lsci"
    OCT = "oct"
    HYBRID = "hybrid"


class DefectType(Enum):
    """Enumeration for different defect types."""
    SURFACE_SCRATCH = "surface_scratch"
    INTERNAL_CRACK = "internal_crack"
    MATERIAL_INCLUSION = "material_inclusion"
    THERMAL_DAMAGE = "thermal_damage"
    CORROSION = "corrosion"
    WEAR = "wear"


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""
    
    # General settings
    num_train_samples: int = 5000
    num_val_samples: int = 1000
    image_size: Tuple[int, int] = (512, 512)
    random_seed: int = 42
    
    # LSCI parameters
    lsci_wavelength: float = 785.0  # nm
    lsci_coherence_length: float = 50.0  # μm
    lsci_speckle_size: float = 2.0  # μm
    lsci_contrast_range: Tuple[float, float] = (0.1, 0.8)
    
    # OCT parameters
    oct_center_wavelength: float = 1300.0  # nm
    oct_bandwidth: float = 100.0  # nm
    oct_axial_resolution: float = 5.0  # μm
    oct_lateral_resolution: float = 10.0  # μm
    oct_depth_range: float = 2000.0  # μm
    
    # Defect generation
    defect_probability: float = 0.3
    defect_size_range: Tuple[float, float] = (0.05, 0.2)  # relative to image size
    defect_intensity_range: Tuple[float, float] = (0.2, 0.8)
    
    # Noise parameters
    thermal_noise_std: float = 0.05
    motion_blur_probability: float = 0.1
    motion_blur_kernel_size: int = 15
    gaussian_noise_std: float = 0.02
    salt_pepper_probability: float = 0.01


@dataclass
class ModelConfig:
    """Configuration for the U-Net model."""
    
    # Architecture
    input_channels: int = 3
    output_channels: int = 1
    initial_features: int = 64
    depth: int = 5
    dropout_rate: float = 0.2
    batch_norm: bool = True
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    
    # Loss function
    loss_type: str = "dice_bce"
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Data augmentation
    augmentation_probability: float = 0.8
    rotation_range: Tuple[float, float] = (-30, 30)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip: bool = True
    vertical_flip: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 5
    eval_interval: int = 1
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: str = "auto-optical-diagnostics"
    
    # Checkpointing
    save_best_only: bool = True
    save_last_checkpoint: bool = True
    max_checkpoints: int = 5
    
    # Validation
    val_split: float = 0.2
    cross_validation_folds: int = 5


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "dice_score", "iou", "precision", "recall", "f1_score",
        "hausdorff_distance", "surface_dice"
    ])
    
    # Threshold optimization
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    threshold_steps: int = 50
    
    # Visualization
    num_visualization_samples: int = 20
    save_predictions: bool = True
    save_attention_maps: bool = True
    save_gradcam: bool = True
    
    # Statistical analysis
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95, 0.99])
    bootstrap_samples: int = 1000


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    
    # Base paths
    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")
    
    # Data paths
    synthetic_data_dir: Path = Path("data/synthetic")
    train_images_dir: Path = Path("data/synthetic/train/images")
    train_masks_dir: Path = Path("data/synthetic/train/masks")
    val_images_dir: Path = Path("data/synthetic/val/images")
    val_masks_dir: Path = Path("data/synthetic/val/masks")
    
    # Model paths
    saved_models_dir: Path = Path("models/saved_models")
    checkpoints_dir: Path = Path("models/checkpoints")
    
    # Output paths
    evaluation_results_dir: Path = Path("outputs/evaluation_results")
    visualizations_dir: Path = Path("outputs/visualizations")
    reports_dir: Path = Path("outputs/reports")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in self.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class that combines all sub-configurations."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from file or defaults."""
        self.data_generation = DataGenerationConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.paths = PathsConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'data_generation' in config_dict:
            for key, value in config_dict['data_generation'].items():
                setattr(self.data_generation, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(self.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(self.training, key, value)
        
        if 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                setattr(self.evaluation, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data_generation': self.data_generation.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'paths': str(self.paths.base_dir)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_device(self) -> str:
        """Get the appropriate device for training."""
        if self.training.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.training.device
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate data generation config
        if self.data_generation.num_train_samples <= 0:
            errors.append("num_train_samples must be positive")
        
        if self.data_generation.image_size[0] <= 0 or self.data_generation.image_size[1] <= 0:
            errors.append("image_size must be positive")
        
        # Validate model config
        if self.model.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.model.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Validate training config
        if self.training.val_split < 0 or self.training.val_split > 1:
            errors.append("val_split must be between 0 and 1")
        
        return errors


# Global configuration instance
config = Config()