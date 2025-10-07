# Bio-Inspired Navigation Data Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready Python package for generating and visualizing bio-inspired navigation research data. This project provides sophisticated data simulation, high-quality visualization, and robust analysis tools for bio-inspired navigation research.

## 🚀 Features

- **📊 Sophisticated Data Simulation**: Advanced trajectory generation with periodic error correction
- **⚡ Energy Consumption Analysis**: Realistic energy consumption modeling with statistical validation
- **📈 High-Quality Visualizations**: Publication-ready plots with customizable styling
- **🔧 Modular Architecture**: Clean, maintainable code structure with proper separation of concerns
- **📝 Comprehensive Logging**: Detailed logging and error tracking throughout the pipeline
- **⚙️ Configuration Management**: Centralized configuration with validation
- **🧪 Data Validation**: Built-in data quality checks and statistical analysis
- **📦 Production Ready**: Proper packaging, error handling, and documentation
- **🤖 ROS2 Integration**: Optional ROS2 node for real-time data streaming and research prototyping
- **🐳 Containerized Deployment**: Ready-to-use Docker and Compose setup for reproducible runs

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/bio-nav-data.git
cd bio-nav-data

# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run the application to verify everything works
python main.py --config-only
```

## 🚀 Quick Start

### Basic Usage

```bash
# Generate all data and plots
python main.py

# Show configuration
python main.py --config-only

# Validate setup
python main.py --validate
```

### Programmatic Usage

```python
from bio_nav_data import BioNavDataGenerator, Config

# Initialize with custom configuration
config = Config()
generator = BioNavDataGenerator(config)

# Run complete pipeline
generator.run_complete_pipeline()
```

## 📖 Usage

### Command Line Interface

The application provides a comprehensive command-line interface:

```bash
# Basic usage - generate all data and plots
python main.py

# Show configuration details
python main.py --config-only

# Validate configuration
python main.py --validate

# Set custom output directory
python main.py --output-dir /path/to/output

# Set logging level
python main.py --log-level DEBUG

# Get help
python main.py --help
```

### Configuration Options

The application uses a centralized configuration system:

```python
from bio_nav_data.utils.config import Config

# Create custom configuration
config = Config(base_path="/custom/path")

# Update parameters
config.update_trajectory_params(n_points=200, distance_m=150)
config.update_energy_params(n_trials=150)
config.update_plot_params(dpi=600)

# Validate configuration
validation = config.validate_config()
```

### Data Generation

Generate different types of research data:

```python
from bio_nav_data.data_generators import (
    TrajectoryDataGenerator, 
    EnergyDataGenerator,
    create_performance_summary
)

# Generate trajectory data
traj_gen = TrajectoryDataGenerator(n_points=101, distance_m=100)
trajectory_df = traj_gen.generate()

# Generate energy consumption data
energy_gen = EnergyDataGenerator(n_trials=100)
energy_df = energy_gen.generate()

# Generate performance summary
performance_df = create_performance_summary()
```

### Visualization

Create publication-ready plots:

```python
from bio_nav_data.visualizers.plots import (
    plot_trajectory,
    plot_energy_consumption,
    plot_performance_comparison
)

# Create trajectory plot
plot_trajectory(trajectory_df, save_path="trajectory.png")

# Create energy consumption plot
plot_energy_consumption(energy_df, save_path="energy.png")

# Create performance comparison
plot_performance_comparison(performance_df, save_path="performance.png")
```

## 📁 Project Structure

```
bio_nav_data/
├── bio_nav_data/                 # Main package
│   ├── __init__.py              # Package initialization
│   ├── data_generators/         # Data generation modules
│   │   ├── __init__.py
│   │   ├── trajectory.py        # Trajectory data generator
│   │   ├── energy.py            # Energy consumption generator
│   │   └── performance.py       # Performance metrics generator
│   ├── visualizers/             # Visualization modules
│   │   ├── __init__.py
│   │   └── plots.py             # Plotting functions
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
├── main.py                      # Main application script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── output/                      # Generated data files
├── plots/                       # Generated plots
└── logs/                        # Application logs
```

## ⚙️ Configuration

### Default Configuration

The application comes with sensible defaults:

```python
# Trajectory parameters
trajectory_params = {
    'n_points': 101,              # Number of trajectory points
    'distance_m': 100.0,          # Total distance in meters
    'correction_interval': 20,    # Error correction interval
    'random_seed': 42             # Random seed for reproducibility
}

# Energy parameters
energy_params = {
    'n_trials': 100,              # Number of trials per method
    'random_seed': 42             # Random seed for reproducibility
}

# Plot parameters
plot_params = {
    'dpi': 300,                   # Plot resolution
    'figsize': (10, 8),          # Figure size
    'save_format': 'png',         # Output format
    'style': 'seaborn-v0_8-whitegrid'  # Plot style
}
```

### Custom Configuration

You can customize any aspect of the configuration:

```python
from bio_nav_data.utils.config import Config

config = Config()

# Update trajectory parameters
config.update_trajectory_params(
    n_points=200,
    distance_m=150,
    correction_interval=30
)

# Update energy parameters
config.update_energy_params(n_trials=200)

# Update plot parameters
config.update_plot_params(dpi=600, figsize=(12, 10))
```

## 📚 API Documentation

### Core Classes

#### `BioNavDataGenerator`

Main application class for orchestrating data generation and visualization.

```python
class BioNavDataGenerator:
    def __init__(self, config: Optional[Config] = None)
    def generate_all_data(self) -> Dict[str, Any]
    def save_all_data(self, data: Dict[str, Any]) -> None
    def generate_all_plots(self, data: Dict[str, Any]) -> None
    def run_complete_pipeline(self) -> None
```

#### `TrajectoryDataGenerator`

Generates sophisticated trajectory data with drift simulation and error correction.

```python
class TrajectoryDataGenerator:
    def __init__(self, n_points: int = 101, distance_m: float = 100.0, 
                 correction_interval: int = 20, random_seed: int = 42)
    def generate(self) -> pd.DataFrame
    def get_statistics(self, df: pd.DataFrame) -> dict
```

#### `EnergyDataGenerator`

Generates realistic energy consumption data with statistical validation.

```python
class EnergyDataGenerator:
    def __init__(self, n_trials: int = 100, random_seed: int = 42)
    def generate(self) -> pd.DataFrame
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]
    def calculate_improvements(self, df: pd.DataFrame) -> Dict[str, float]
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, bool]
```

### Visualization Functions

#### `plot_trajectory()`

Creates high-quality trajectory plots.

```python
def plot_trajectory(df: pd.DataFrame, 
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   dpi: int = 300,
                   show_grid: bool = True,
                   show_legend: bool = True) -> plt.Figure
```

#### `plot_energy_consumption()`

Creates energy consumption box plots with statistical annotations.

```python
def plot_energy_consumption(df: pd.DataFrame,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          dpi: int = 300,
                          show_outliers: bool = True,
                          show_statistics: bool = True) -> plt.Figure
```

## 📊 Examples

### Example 1: Basic Data Generation

```python
from bio_nav_data import BioNavDataGenerator

# Create generator
generator = BioNavDataGenerator()

# Generate all data
data = generator.generate_all_data()

# Access individual datasets
trajectory_df = data['trajectory']
energy_df = data['energy']
performance_df = data['performance']

print(f"Generated {len(trajectory_df)} trajectory points")
print(f"Generated {len(energy_df)} energy measurements")
print(f"Generated {len(performance_df)} performance metrics")
```

### Example 2: Custom Configuration

```python
from bio_nav_data import BioNavDataGenerator
from bio_nav_data.utils.config import Config

# Create custom configuration
config = Config()
config.update_trajectory_params(n_points=200, distance_m=150)
config.update_energy_params(n_trials=150)
config.update_plot_params(dpi=600)

# Use custom configuration
generator = BioNavDataGenerator(config)
generator.run_complete_pipeline()
```

### Example 3: Individual Components

```python
from bio_nav_data.data_generators import TrajectoryDataGenerator
from bio_nav_data.visualizers.plots import plot_trajectory

# Generate trajectory data
traj_gen = TrajectoryDataGenerator(n_points=101, distance_m=100)
trajectory_df = traj_gen.generate()

# Get statistics
stats = traj_gen.get_statistics(trajectory_df)
print(f"Traditional SLAM drift: {stats['traditional_slam_avg_drift']:.3f}m")
print(f"Our framework error: {stats['our_framework_avg_error']:.3f}m")
print(f"Improvement ratio: {stats['improvement_ratio']:.2f}x")

# Create plot
plot_trajectory(trajectory_df, save_path="custom_trajectory.png")
```

### Example 4: Data Analysis

```python
from bio_nav_data.data_generators import EnergyDataGenerator

# Generate energy data
energy_gen = EnergyDataGenerator(n_trials=100)
energy_df = energy_gen.generate()

# Get comprehensive statistics
stats = energy_gen.get_statistics(energy_df)
improvements = energy_gen.calculate_improvements(energy_df)
validation = energy_gen.validate_data_quality(energy_df)

# Print results
for method, method_stats in stats.items():
    print(f"{method}:")
    print(f"  Mean: {method_stats['mean']:.2f}W")
    print(f"  Std: {method_stats['std']:.2f}W")
    print(f"  Median: {method_stats['median']:.2f}W")

print(f"\nImprovements:")
for baseline, improvement in improvements.items():
    print(f"  {baseline}: {improvement:.1f}% reduction")

print(f"\nData Quality: {all(validation.values())}")
```

## 🐳 Docker Usage

Run the complete pipeline without installing local dependencies by using the provided container assets.

### Build the Image

```bash
docker build -t bio-nav-data:latest .
```

### Run Once

```bash
docker run --rm -v "$(pwd)/output:/app/output" -v "$(pwd)/plots:/app/plots" -v "$(pwd)/logs:/app/logs" bio-nav-data:latest --log-level INFO
```

### Using Docker Compose

```bash
LOG_LEVEL=DEBUG docker compose up --build
```

The Compose stack mounts the local `output`, `plots`, and `logs` folders so generated artifacts persist on the host. Override CLI arguments by editing `docker-compose.yml` or passing a different `LOG_LEVEL` environment variable at runtime.

## 🤖 ROS2 Integration

An optional ROS2 package is included under `ros2/src/bio_nav_data_ros`. It wraps the
Python pipeline in a ROS2 node for real-time experimentation and visualization.

### Build & Launch

1. Install ROS2 (Humble or newer) and ensure `colcon` is available.
   - macOS/Linux: install ROS2 from the official instructions, then `pip install colcon-common-extensions` if needed.
   - Source the ROS2 environment before building: `source /opt/ros/humble/setup.bash` (path may vary).
2. Install the core Python package so the node can import the pipeline:
   ```bash
   pip install -e .
   ```
3. Build the ROS2 workspace and source the overlay:
   ```bash
   cd ros2
   colcon build --packages-select bio_nav_data_ros
   source install/setup.bash
   ```
4. Launch the node:
   ```bash
   ros2 launch bio_nav_data_ros bio_nav_data.launch.py
   ```

### Published Interfaces

- Trajectory paths: `bio_nav_data/trajectory/path/{ground_truth, our_framework, traditional_slam}` (`nav_msgs/Path`)
- Data snapshots: `bio_nav_data/trajectory/json`, `bio_nav_data/energy/json`, `bio_nav_data/performance/json` (`std_msgs/String`, JSON payloads)
- Analytics summary: `bio_nav_data/analytics/json` (`std_msgs/String`, statistics & validation results)
- Trigger new run: `bio_nav_data/generate` (`std_srvs/Trigger`)

### Node Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `publish_frequency_hz` | double | `1.0` | Rate for republishing cached messages |
| `frame_id` | string | `map` | Coordinate frame stamped on `nav_msgs/Path` messages |
| `auto_generate_on_startup` | bool | `true` | Run the pipeline automatically when the node starts |
| `output_directory` | string | `` | Optional override for generated file locations |
| `save_results_to_disk` | bool | `false` | Persist CSV outputs on each run |
| `generate_plots` | bool | `false` | Produce plots alongside CSV outputs |
| `trajectory.*` | mixed | defaults from `Config` | Fine-tune trajectory generator settings |
| `energy.*` | mixed | defaults from `Config` | Fine-tune energy generator settings |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the coding standards
4. **Add tests**: Ensure your code is well-tested
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes clearly

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bio-nav-data.git
cd bio-nav-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Check code quality
flake8
```

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings
- Add logging for important operations
- Include error handling for robust operation
- Write unit tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research team for the original bio-inspired navigation concepts
- Open source community for the excellent Python libraries used
- Contributors and reviewers for their valuable feedback

## 📞 Support

If you encounter any issues or have questions:

1. Check the [documentation](#api-documentation)
2. Search existing [issues](https://github.com/yourusername/bio-nav-data/issues)
3. Create a new issue with detailed information
4. Contact the development team


**Made with ❤️ for the research community**
