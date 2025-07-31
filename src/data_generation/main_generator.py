"""
Main data generator for AutoOpticalDiagnostics.
Orchestrates the generation of synthetic optical imaging data with defects.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

from .lsci_simulator import LSCISimulator, create_lsci_config
from .oct_simulator import OCTSimulator, create_oct_config
from .noise_models import NoiseModel, AdvancedNoiseModel, create_noise_profile
from ..config import DefectType, ImagingMode
from ..utils import setup_logging, create_visualization_grid


class DataGenerator:
    """
    Main data generator for synthetic optical imaging data.
    Coordinates LSCI and OCT simulators with defect generation and noise models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize simulators
        self.lsci_simulator = LSCISimulator(config)
        self.oct_simulator = OCTSimulator(config)
        
        # Initialize noise models
        self.noise_model = AdvancedNoiseModel(config)
        
        # Data generation parameters
        self.image_size = config.get('image_size', (512, 512))
        self.num_train_samples = config.get('num_train_samples', 5000)
        self.num_val_samples = config.get('num_val_samples', 1000)
        self.defect_probability = config.get('defect_probability', 0.3)
        self.defect_size_range = config.get('defect_size_range', (0.05, 0.2))
        self.defect_intensity_range = config.get('defect_intensity_range', (0.2, 0.8))
        
        # Output paths
        self.output_paths = {
            'train_images': Path('data/synthetic/train/images'),
            'train_masks': Path('data/synthetic/train/masks'),
            'val_images': Path('data/synthetic/val/images'),
            'val_masks': Path('data/synthetic/val/masks'),
            'metadata': Path('data/synthetic/metadata')
        }
        
        # Create output directories
        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Defect types and their probabilities
        self.defect_types = list(DefectType)
        self.defect_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Probabilities for each defect type
        
        # Imaging modes
        self.imaging_modes = list(ImagingMode)
        
    def generate_dataset(self, dataset_type: str = "train") -> None:
        """
        Generate complete dataset (train or validation).
        
        Args:
            dataset_type: Type of dataset ('train' or 'val')
        """
        self.logger.info(f"Generating {dataset_type} dataset...")
        
        if dataset_type == "train":
            num_samples = self.num_train_samples
            image_path = self.output_paths['train_images']
            mask_path = self.output_paths['train_masks']
        else:
            num_samples = self.num_val_samples
            image_path = self.output_paths['val_images']
            mask_path = self.output_paths['val_masks']
        
        # Generate samples
        metadata = []
        
        for i in tqdm(range(num_samples), desc=f"Generating {dataset_type} samples"):
            # Generate single sample
            sample_metadata = self._generate_sample(i, image_path, mask_path)
            metadata.append(sample_metadata)
        
        # Save metadata
        metadata_path = self.output_paths['metadata'] / f"{dataset_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Generated {num_samples} {dataset_type} samples")
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def _generate_sample(
        self, 
        sample_id: int, 
        image_path: Path, 
        mask_path: Path
    ) -> Dict[str, Any]:
        """
        Generate a single sample with image and mask.
        
        Args:
            sample_id: Sample identifier
            image_path: Path to save image
            mask_path: Path to save mask
            
        Returns:
            Sample metadata
        """
        # Randomly choose imaging mode
        imaging_mode = random.choices(self.imaging_modes, weights=[0.4, 0.4, 0.2])[0]
        
        # Decide if sample has defect
        has_defect = random.random() < self.defect_probability
        
        # Generate defect mask if needed
        defect_mask = None
        defect_type = None
        defect_properties = None
        
        if has_defect:
            defect_type = random.choices(self.defect_types, weights=self.defect_weights)[0]
            defect_mask, defect_properties = self._generate_defect_mask(defect_type)
        
        # Generate optical image based on mode
        if imaging_mode == ImagingMode.LSCI:
            optical_image = self._generate_lsci_image(defect_mask, defect_type)
        elif imaging_mode == ImagingMode.OCT:
            optical_image = self._generate_oct_image(defect_mask, defect_type)
        else:  # HYBRID
            optical_image = self._generate_hybrid_image(defect_mask, defect_type)
        
        # Apply noise
        noisy_image = self.noise_model.apply_comprehensive_noise(optical_image)
        
        # Create binary mask for defect detection
        binary_mask = np.zeros(self.image_size, dtype=np.uint8)
        if defect_mask is not None:
            binary_mask[defect_mask > 0.5] = 255
        
        # Save image and mask
        image_filename = f"sample_{sample_id:06d}.png"
        mask_filename = f"sample_{sample_id:06d}_mask.png"
        
        # Convert to PIL and save
        image_pil = Image.fromarray((noisy_image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(binary_mask)
        
        image_pil.save(image_path / image_filename)
        mask_pil.save(mask_path / mask_filename)
        
        # Create metadata
        metadata = {
            'sample_id': sample_id,
            'filename': image_filename,
            'mask_filename': mask_filename,
            'imaging_mode': imaging_mode.value,
            'has_defect': has_defect,
            'defect_type': defect_type.value if defect_type else None,
            'defect_properties': defect_properties,
            'image_size': self.image_size,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    def _generate_defect_mask(
        self, 
        defect_type: DefectType
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate defect mask for given defect type.
        
        Args:
            defect_type: Type of defect to generate
            
        Returns:
            Tuple of (defect_mask, defect_properties)
        """
        height, width = self.image_size
        defect_mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Random defect size
        defect_size = random.uniform(*self.defect_size_range)
        defect_pixels = int(min(height, width) * defect_size)
        
        # Random defect position
        center_x = random.randint(defect_pixels//2, width - defect_pixels//2)
        center_y = random.randint(defect_pixels//2, height - defect_pixels//2)
        
        defect_properties = {
            'type': defect_type.value,
            'size': defect_size,
            'center': (center_x, center_y),
            'intensity': random.uniform(*self.defect_intensity_range)
        }
        
        if defect_type == DefectType.SURFACE_SCRATCH:
            defect_mask = self._create_surface_scratch_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        elif defect_type == DefectType.INTERNAL_CRACK:
            defect_mask = self._create_internal_crack_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        elif defect_type == DefectType.MATERIAL_INCLUSION:
            defect_mask = self._create_material_inclusion_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        elif defect_type == DefectType.THERMAL_DAMAGE:
            defect_mask = self._create_thermal_damage_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        elif defect_type == DefectType.CORROSION:
            defect_mask = self._create_corrosion_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        elif defect_type == DefectType.WEAR:
            defect_mask = self._create_wear_mask(
                center_x, center_y, defect_pixels, defect_properties
            )
        
        return defect_mask, defect_properties
    
    def _create_surface_scratch_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create surface scratch defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create linear scratch
        angle = random.uniform(0, 2 * np.pi)
        length = size
        width = max(1, size // 10)
        
        # Create scratch line
        x1 = int(center_x - length//2 * np.cos(angle))
        y1 = int(center_y - length//2 * np.sin(angle))
        x2 = int(center_x + length//2 * np.cos(angle))
        y2 = int(center_y + length//2 * np.sin(angle))
        
        cv2.line(mask, (x1, y1), (x2, y2), properties['intensity'], width)
        
        # Add some roughness
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    def _create_internal_crack_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create internal crack defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create branching crack pattern
        num_branches = random.randint(2, 4)
        main_length = size // 2
        
        # Main crack
        angle = random.uniform(0, 2 * np.pi)
        x1 = int(center_x - main_length * np.cos(angle))
        y1 = int(center_y - main_length * np.sin(angle))
        x2 = int(center_x + main_length * np.cos(angle))
        y2 = int(center_y + main_length * np.sin(angle))
        
        cv2.line(mask, (x1, y1), (x2, y2), properties['intensity'], 2)
        
        # Add branches
        for _ in range(num_branches):
            branch_angle = angle + random.uniform(-np.pi/4, np.pi/4)
            branch_length = random.randint(size//4, size//2)
            branch_x = random.randint(x1, x2)
            branch_y = random.randint(y1, y2)
            
            bx1 = int(branch_x - branch_length * np.cos(branch_angle))
            by1 = int(branch_y - branch_length * np.sin(branch_angle))
            bx2 = int(branch_x + branch_length * np.cos(branch_angle))
            by2 = int(branch_y + branch_length * np.sin(branch_angle))
            
            cv2.line(mask, (bx1, by1), (bx2, by2), properties['intensity'], 1)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    def _create_material_inclusion_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create material inclusion defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create irregular inclusion shape
        shape_type = random.choice(['circle', 'ellipse', 'polygon'])
        
        if shape_type == 'circle':
            radius = size // 2
            cv2.circle(mask, (center_x, center_y), radius, properties['intensity'], -1)
        
        elif shape_type == 'ellipse':
            axes = (size//2, size//3)
            angle = random.uniform(0, 180)
            cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, properties['intensity'], -1)
        
        else:  # polygon
            num_vertices = random.randint(5, 8)
            vertices = []
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                radius = size//2 * random.uniform(0.7, 1.0)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                vertices.append([x, y])
            
            vertices = np.array(vertices, dtype=np.int32)
            cv2.fillPoly(mask, [vertices], properties['intensity'])
        
        # Add some texture
        noise = np.random.normal(0, 0.1, mask.shape)
        mask = np.clip(mask + noise, 0, 1)
        
        return mask
    
    def _create_thermal_damage_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create thermal damage defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create thermal damage pattern
        radius = size // 2
        
        # Create radial damage pattern
        y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Radial intensity falloff
        intensity = properties['intensity'] * np.exp(-distance / (radius * 0.5))
        mask[distance <= radius] = intensity[distance <= radius]
        
        # Add thermal stress patterns
        stress_pattern = np.random.normal(0, 0.2, mask.shape)
        mask = np.clip(mask + stress_pattern, 0, 1)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def _create_corrosion_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create corrosion defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create corrosion pattern
        radius = size // 2
        
        # Create irregular corrosion boundary
        y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Add irregularity to boundary
        angle = np.arctan2(y - center_y, x - center_x)
        irregular_radius = radius * (1 + 0.3 * np.sin(3 * angle) + 0.2 * np.cos(5 * angle))
        
        # Create corrosion mask
        corrosion_region = distance <= irregular_radius
        mask[corrosion_region] = properties['intensity']
        
        # Add corrosion texture
        texture = np.random.normal(0, 0.15, mask.shape)
        mask = np.clip(mask + texture, 0, 1)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _create_wear_mask(
        self, 
        center_x: int, 
        center_y: int, 
        size: int, 
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Create wear defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Create wear pattern
        radius = size // 2
        
        # Create directional wear
        wear_direction = random.uniform(0, 2 * np.pi)
        wear_length = size
        wear_width = size // 3
        
        # Create elliptical wear pattern
        axes = (wear_length//2, wear_width//2)
        cv2.ellipse(mask, (center_x, center_y), axes, np.degrees(wear_direction), 0, 360, properties['intensity'], -1)
        
        # Add wear gradient
        y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        gradient = 1 - (distance / radius)
        gradient = np.clip(gradient, 0, 1)
        
        mask = mask * gradient
        
        # Add wear texture
        texture = np.random.normal(0, 0.1, mask.shape)
        mask = np.clip(mask + texture, 0, 1)
        
        return mask
    
    def _generate_lsci_image(
        self, 
        defect_mask: Optional[np.ndarray], 
        defect_type: Optional[DefectType]
    ) -> np.ndarray:
        """Generate LSCI image."""
        # Generate base speckle pattern
        speckle_pattern = self.lsci_simulator.generate_speckle_pattern(self.image_size)
        
        # Apply defect if present
        if defect_mask is not None and defect_type is not None:
            speckle_pattern = self.lsci_simulator.generate_defect_speckle(
                speckle_pattern, defect_mask, defect_type.value
            )
        
        # Calculate speckle contrast
        contrast_image = self.lsci_simulator.calculate_speckle_contrast(speckle_pattern)
        
        # Convert to RGB
        rgb_image = np.stack([contrast_image] * 3, axis=2)
        
        return rgb_image
    
    def _generate_oct_image(
        self, 
        defect_mask: Optional[np.ndarray], 
        defect_type: Optional[DefectType]
    ) -> np.ndarray:
        """Generate OCT image."""
        # Generate OCT image
        oct_image = self.oct_simulator.generate_oct_image(self.image_size)
        
        # Apply defect if present
        if defect_mask is not None and defect_type is not None:
            # Convert 2D defect mask to 1D for OCT
            defect_mask_1d = np.mean(defect_mask, axis=0)
            oct_image = self.oct_simulator.generate_defect_oct(
                oct_image, defect_mask_1d, defect_type.value
            )
        
        # Convert to RGB (repeat the same channel)
        rgb_image = np.stack([oct_image] * 3, axis=2)
        
        return rgb_image
    
    def _generate_hybrid_image(
        self, 
        defect_mask: Optional[np.ndarray], 
        defect_type: Optional[DefectType]
    ) -> np.ndarray:
        """Generate hybrid LSCI-OCT image."""
        # Generate both LSCI and OCT images
        lsci_image = self._generate_lsci_image(defect_mask, defect_type)
        oct_image = self._generate_oct_image(defect_mask, defect_type)
        
        # Combine images (LSCI in red channel, OCT in green channel)
        hybrid_image = np.zeros((*self.image_size, 3))
        hybrid_image[:, :, 0] = lsci_image[:, :, 0]  # Red channel (LSCI)
        hybrid_image[:, :, 1] = oct_image[:, :, 0]   # Green channel (OCT)
        hybrid_image[:, :, 2] = (lsci_image[:, :, 0] + oct_image[:, :, 0]) / 2  # Blue channel (combined)
        
        return hybrid_image
    
    def generate_sample_visualization(self, num_samples: int = 16) -> None:
        """
        Generate visualization of sample images and masks.
        
        Args:
            num_samples: Number of samples to visualize
        """
        self.logger.info("Generating sample visualization...")
        
        # Generate sample images
        images = []
        masks = []
        titles = []
        
        for i in range(num_samples):
            # Generate sample
            defect_mask = None
            defect_type = None
            
            if random.random() < self.defect_probability:
                defect_type = random.choice(self.defect_types)
                defect_mask, _ = self._generate_defect_mask(defect_type)
            
            # Generate image
            imaging_mode = random.choice(self.imaging_modes)
            if imaging_mode == ImagingMode.LSCI:
                image = self._generate_lsci_image(defect_mask, defect_type)
            elif imaging_mode == ImagingMode.OCT:
                image = self._generate_oct_image(defect_mask, defect_type)
            else:
                image = self._generate_hybrid_image(defect_mask, defect_type)
            
            # Apply noise
            noisy_image = self.noise_model.apply_comprehensive_noise(image)
            
            # Create mask
            binary_mask = np.zeros(self.image_size, dtype=np.uint8)
            if defect_mask is not None:
                binary_mask[defect_mask > 0.5] = 255
            
            images.append(noisy_image)
            masks.append(binary_mask)
            
            defect_name = defect_type.value if defect_type else "No Defect"
            titles.append(f"{imaging_mode.value} - {defect_name}")
        
        # Create visualization
        from ..utils import create_visualization_grid
        
        # Save image grid
        create_visualization_grid(
            images, titles, save_path="outputs/visualizations/sample_images.png"
        )
        
        # Save mask grid
        create_visualization_grid(
            masks, titles, save_path="outputs/visualizations/sample_masks.png"
        )
        
        self.logger.info("Sample visualization saved to outputs/visualizations/")
    
    def generate_dataset_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the generated dataset.
        
        Returns:
            Dataset statistics
        """
        self.logger.info("Generating dataset statistics...")
        
        # Load metadata
        train_metadata_path = self.output_paths['metadata'] / "train_metadata.json"
        val_metadata_path = self.output_paths['metadata'] / "val_metadata.json"
        
        train_metadata = []
        val_metadata = []
        
        if train_metadata_path.exists():
            with open(train_metadata_path, 'r') as f:
                train_metadata = json.load(f)
        
        if val_metadata_path.exists():
            with open(val_metadata_path, 'r') as f:
                val_metadata = json.load(f)
        
        # Calculate statistics
        stats = {
            'total_samples': len(train_metadata) + len(val_metadata),
            'train_samples': len(train_metadata),
            'val_samples': len(val_metadata),
            'defect_distribution': {},
            'imaging_mode_distribution': {},
            'defect_rate': 0.0
        }
        
        all_metadata = train_metadata + val_metadata
        
        if all_metadata:
            # Defect distribution
            defect_counts = {}
            imaging_mode_counts = {}
            defect_samples = 0
            
            for sample in all_metadata:
                if sample['has_defect']:
                    defect_samples += 1
                    defect_type = sample['defect_type']
                    defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                
                imaging_mode = sample['imaging_mode']
                imaging_mode_counts[imaging_mode] = imaging_mode_counts.get(imaging_mode, 0) + 1
            
            stats['defect_distribution'] = defect_counts
            stats['imaging_mode_distribution'] = imaging_mode_counts
            stats['defect_rate'] = defect_samples / len(all_metadata)
        
        # Save statistics
        stats_path = self.output_paths['metadata'] / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset statistics saved to {stats_path}")
        return stats


def create_data_generator_config(
    num_train_samples: int = 5000,
    num_val_samples: int = 1000,
    image_size: Tuple[int, int] = (512, 512),
    defect_probability: float = 0.3
) -> Dict[str, Any]:
    """
    Create data generator configuration.
    
    Args:
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        image_size: Image size
        defect_probability: Probability of defect occurrence
        
    Returns:
        Data generator configuration
    """
    config = {
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'image_size': image_size,
        'defect_probability': defect_probability,
        'defect_size_range': (0.05, 0.2),
        'defect_intensity_range': (0.2, 0.8),
        'random_seed': 42
    }
    
    # Add LSCI configuration
    config.update(create_lsci_config())
    
    # Add OCT configuration
    config.update(create_oct_config())
    
    # Add noise configuration
    config.update(create_noise_profile('industrial'))
    
    return config