"""
Configuration file for AutoOpticalDiagnostics project.
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
    BOUNDARY_DEFECT = "boundary_defect"
    THERMAL_DAMAGE = "thermal_damage"
    STRESS_CONCENTRATION = "stress_concentration"

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
    defect_size_range: Tuple[float, float] = (0.1, 0.3)  # relative to image size
    defect_intensity_range: Tuple[float, float] = (0.2, 0.8)
    
    # Noise parameters
    thermal_noise_std: float = 0.05
    motion_blur_probability: float = 0.1
    motion_blur_kernel_size: int = 15
    vibration_amplitude: float = 2.0  # pixels
    
    # Material properties
    materials: List[str] = field(default_factory=lambda: [
        "aluminum", "steel", "titanium", "ceramic", "composite"
    ])
    material_properties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "aluminum": {"reflectivity": 0.85, "roughness": 0.1, "thermal_conductivity": 237.0},
        "steel": {"reflectivity": 0.75, "roughness": 0.15, "thermal_conductivity": 50.0},
        "titanium": {"reflectivity": 0.65, "roughness": 0.2, "thermal_conductivity": 22.0},
        "ceramic": {"reflectivity": 0.4, "roughness": 0.05, "thermal_conductivity": 30.0},
        "composite": {"reflectivity": 0.6, "roughness": 0.12, "thermal_conductivity": 5.0}
    })

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
    attention_gates: bool = True
    residual_connections: bool = True
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    
    # Loss function
    loss_type: str = "dice_bce"  # "dice", "bce", "dice_bce", "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_smooth: float = 1e-6
    
    # Data augmentation
    augmentation_probability: float = 0.8
    rotation_range: float = 30.0
    scale_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
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
    validation_split: float = 0.2
    cross_validation_folds: int = 5

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "dice_score", "iou", "precision", "recall", "f1_score", 
        "hausdorff_distance", "surface_dice", "volume_dice"
    ])
    
    # Threshold optimization
    threshold_range: Tuple[float, float] = (0.1, 0.9)
    threshold_steps: int = 50
    
    # Visualization
    num_visualization_samples: int = 20
    save_predictions: bool = True
    create_animation: bool = True
    
    # Statistical analysis
    confidence_intervals: bool = True
    bootstrap_samples: int = 1000
    statistical_tests: bool = True

@dataclass
class PathsConfig:
    """Configuration for file paths."""
    
    # Base directories
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.outputs_dir = self.base_dir / "outputs"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories if they don't exist
        for path in [self.data_dir, self.models_dir, self.outputs_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @property
    def synthetic_data_dir(self) -> Path:
        return self.data_dir / "synthetic"
    
    @property
    def train_images_dir(self) -> Path:
        return self.synthetic_data_dir / "train" / "images"
    
    @property
    def train_masks_dir(self) -> Path:
        return self.synthetic_data_dir / "train" / "masks"
    
    @property
    def val_images_dir(self) -> Path:
        return self.synthetic_data_dir / "val" / "images"
    
    @property
    def val_masks_dir(self) -> Path:
        return self.synthetic_data_dir / "val" / "masks"
    
    @property
    def saved_models_dir(self) -> Path:
        return self.models_dir / "saved_models"
    
    @property
    def evaluation_results_dir(self) -> Path:
        return self.outputs_dir / "evaluation_results"

@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    # Global settings
    project_name: str = "AutoOpticalDiagnostics"
    version: str = "1.0.0"
    description: str = "Advanced AI-driven optical diagnostics for industrial quality control"
    
    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self._to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        self._from_dict(config_dict)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_generation': self.data_generation.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'project_name': self.project_name,
            'version': self.version,
            'description': self.description
        }
    
    def _from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
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
        
        self.project_name = config_dict.get('project_name', self.project_name)
        self.version = config_dict.get('version', self.version)
        self.description = config_dict.get('description', self.description)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

# Environment-specific configurations
def get_development_config() -> Config:
    """Get development configuration with reduced dataset sizes."""
    dev_config = Config()
    dev_config.data_generation.num_train_samples = 100
    dev_config.data_generation.num_val_samples = 20
    dev_config.model.num_epochs = 5
    dev_config.training.batch_size = 4
    return dev_config

def get_production_config() -> Config:
    """Get production configuration optimized for performance."""
    prod_config = Config()
    prod_config.data_generation.num_train_samples = 10000
    prod_config.data_generation.num_val_samples = 2000
    prod_config.model.num_epochs = 200
    prod_config.training.batch_size = 16
    prod_config.training.num_workers = 8
    return prod_config