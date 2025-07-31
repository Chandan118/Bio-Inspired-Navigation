# AutoOpticalDiagnostics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced Optical Diagnostics System for Industrial Defect Detection**

AutoOpticalDiagnostics is a comprehensive framework for synthetic data generation and AI-driven defect detection using Laser Speckle Contrast Imaging (LSCI) and Optical Coherence Tomography (OCT). This project implements state-of-the-art deep learning models for automated optical inspection in industrial environments.

## 🚀 Features

### 🔬 Physics-Based Simulation
- **LSCI Simulator**: Realistic speckle pattern generation with configurable parameters
- **OCT Simulator**: Depth-resolved imaging with material property simulation
- **Hybrid Imaging**: Combined LSCI-OCT data generation
- **Industrial Noise Models**: Thermal, motion blur, vibration, and environmental effects

### 🤖 Advanced AI Models
- **U-Net Architectures**: Standard, U-Net++, Attention U-Net, Deep Supervision
- **Multiple Loss Functions**: Dice, BCE, Focal, IoU, Tversky, Boundary, Hausdorff
- **Adaptive Training**: Mixed precision, gradient clipping, early stopping
- **Comprehensive Augmentation**: Realistic industrial condition simulation

### 📊 Defect Types
- Surface scratches and wear patterns
- Internal cracks and material inclusions
- Thermal damage and corrosion
- Delamination and porosity defects

### 🛠️ Professional Features
- **Modular Architecture**: Clean, extensible codebase
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Logging**: Rich console output and file logging
- **Performance Monitoring**: TensorBoard integration and metrics tracking
- **Reproducibility**: Deterministic training with seed management

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoOpticalDiagnostics.git
cd AutoOpticalDiagnostics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### 1. Generate Synthetic Data

```bash
# Generate training and validation datasets
python main.py --mode generate_data --config configs/data_config.yaml

# Generate sample visualization
python main.py --mode visualize --num_samples 16
```

### 2. Train Model

```bash
# Train with default configuration
python main.py --mode train --config configs/training_config.yaml

# Train with custom parameters
python main.py --mode train --config configs/custom_training.yaml --verbose
```

### 3. Evaluate Model

```bash
# Evaluate trained model
python main.py --mode evaluate --model_path outputs/checkpoints/best_checkpoint.pth
```

### 4. Full Pipeline

```bash
# Run complete pipeline (data generation → training → evaluation)
python main.py --mode full_pipeline --config configs/full_config.yaml
```

## 📁 Project Structure

```
AutoOpticalDiagnostics/
├── src/                          # Source code
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── data_generation/          # Synthetic data generation
│   │   ├── lsci_simulator.py     # LSCI simulation
│   │   ├── oct_simulator.py      # OCT simulation
│   │   ├── noise_models.py       # Industrial noise models
│   │   └── main_generator.py     # Data generation orchestration
│   ├── model/                    # Deep learning models
│   │   ├── unet_model.py         # U-Net architectures
│   │   ├── dataset.py            # Data loading and augmentation
│   │   └── loss.py               # Loss functions
│   ├── training/                 # Training framework
│   │   └── train.py              # Advanced trainer
│   └── evaluation/               # Model evaluation
│       └── evaluate.py           # Evaluation metrics
├── data/                         # Data storage
│   └── synthetic/                # Generated synthetic data
├── outputs/                      # Training outputs
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   ├── plots/                    # Training plots
│   └── evaluation_results/       # Evaluation results
├── configs/                      # Configuration files
├── main.py                       # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

### Data Generation Configuration

```yaml
# configs/data_config.yaml
data_generation:
  num_train_samples: 5000
  num_val_samples: 1000
  image_size: [512, 512]
  defect_probability: 0.3
  
  # LSCI parameters
  lsci_wavelength: 785.0
  lsci_speckle_size: 2.0
  lsci_contrast_range: [0.1, 0.8]
  
  # OCT parameters
  oct_center_wavelength: 1300.0
  oct_bandwidth: 100.0
  oct_axial_resolution: 5.0
  
  # Noise parameters
  thermal_noise_std: 0.05
  motion_blur_probability: 0.1
  gaussian_noise_std: 0.02
```

### Training Configuration

```yaml
# configs/training_config.yaml
model:
  model_type: "unet"
  input_channels: 3
  output_channels: 1
  initial_features: 64
  depth: 5
  dropout_rate: 0.2

loss:
  loss_type: "dice_bce"
  dice_weight: 0.5
  bce_weight: 0.5

optimizer:
  type: "adam"
  learning_rate: 1e-4
  weight_decay: 1e-5

data:
  batch_size: 8
  num_workers: 4
  image_size: [512, 512]
  augmentation_probability: 0.8

training:
  num_epochs: 100
  early_stopping_patience: 15
  gradient_clipping: true
  use_amp: true
```

## 📖 Usage Examples

### Custom Data Generation

```python
from src.data_generation import DataGenerator, create_data_generator_config

# Create custom configuration
config = create_data_generator_config(
    num_train_samples=10000,
    num_val_samples=2000,
    image_size=(1024, 1024),
    defect_probability=0.4
)

# Generate data
generator = DataGenerator(config)
generator.generate_dataset("train")
generator.generate_dataset("val")
```

### Custom Model Training

```python
from src.training.train import Trainer, create_training_config

# Create training configuration
config = create_training_config(
    model_type="attention_unet",
    num_epochs=150,
    batch_size=16,
    learning_rate=5e-5,
    loss_type="combined"
)

# Train model
trainer = Trainer(config)
trainer.train()
```

### Model Evaluation

```python
from src.model.unet_model import create_model
import torch

# Load trained model
model = create_model(
    model_type="unet",
    n_channels=3,
    n_classes=1
)

checkpoint = torch.load("outputs/checkpoints/best_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on new data
model.eval()
with torch.no_grad():
    predictions = model(input_images)
```

## 🔧 Advanced Features

### Mixed Precision Training

The framework supports automatic mixed precision (AMP) training for faster training and reduced memory usage:

```python
# Enable in configuration
training:
  use_amp: true
```

### Custom Loss Functions

Create custom loss functions by extending the base classes:

```python
from src.model.loss import DiceLoss

class CustomLoss(DiceLoss):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
    
    def forward(self, input, target):
        # Custom loss computation
        return super().forward(input, target) * self.custom_param
```

### Data Augmentation

Comprehensive augmentation pipeline with industrial-specific transformations:

```python
from src.model.dataset import get_training_transforms

transforms = get_training_transforms(
    image_size=(512, 512),
    augmentation_probability=0.8
)
```

## 📊 Performance Metrics

The framework provides comprehensive evaluation metrics:

- **Segmentation Metrics**: Dice Score, IoU, Precision, Recall, F1-Score
- **Boundary Metrics**: Hausdorff Distance, Boundary Accuracy
- **Quality Metrics**: PSNR, SSIM, Structural Similarity
- **Statistical Analysis**: Confidence intervals, Bootstrap sampling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AutoOpticalDiagnostics.git
cd AutoOpticalDiagnostics

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Albumentations** for image augmentation capabilities
- **Rich** for beautiful console output
- **Research Community** for U-Net and related architectures

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AutoOpticalDiagnostics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AutoOpticalDiagnostics/discussions)
- **Email**: contact@autoopticaldiagnostics.com

## 📈 Citation

If you use this project in your research, please cite:

```bibtex
@article{autoopticaldiagnostics2024,
  title={AutoOpticalDiagnostics: Advanced Optical Diagnostics System for Industrial Defect Detection},
  author={Your Name},
  journal={Journal of Optical Engineering},
  year={2024},
  volume={1},
  number={1},
  pages={1--15}
}
```

---

**Made with ❤️ for the optical diagnostics community**