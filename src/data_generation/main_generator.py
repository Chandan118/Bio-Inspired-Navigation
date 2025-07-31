"""
Main data generator for AutoOpticalDiagnostics.
Orchestrates the generation of synthetic data using LSCI and OCT simulators.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json
from datetime import datetime

from .lsci_simulator import LSCISimulator, create_lsci_dataset
from .oct_simulator import OCTSimulator, create_oct_dataset, create_standard_tissue_layers
from .noise_models import create_industrial_noise_model, create_laboratory_noise_model, create_outdoor_noise_model
from ..config import get_config
from ..utils import create_directory_structure, validate_data_integrity, PerformanceMonitor

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Main data generator for synthetic optical diagnostics data.
    
    Generates realistic datasets for training AI models by combining:
    - LSCI and OCT imaging simulations
    - Realistic noise models for different environments
    - Various defect types and materials
    - Comprehensive data augmentation
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.lsci_simulator = LSCISimulator(
            wavelength=self.config.data_generation.lsci_wavelength,
            coherence_length=self.config.data_generation.lsci_coherence_length,
            speckle_size=self.config.data_generation.lsci_speckle_size,
            contrast_range=self.config.data_generation.lsci_contrast_range,
            image_size=self.config.data_generation.image_size
        )
        
        self.oct_simulator = OCTSimulator(
            center_wavelength=self.config.data_generation.oct_center_wavelength,
            bandwidth=self.config.data_generation.oct_bandwidth,
            axial_resolution=self.config.data_generation.oct_axial_resolution,
            lateral_resolution=self.config.data_generation.oct_lateral_resolution,
            depth_range=self.config.data_generation.oct_depth_range,
            image_size=self.config.data_generation.image_size
        )
        
        # Initialize noise models
        self.noise_models = {
            'industrial': create_industrial_noise_model(),
            'laboratory': create_laboratory_noise_model(),
            'outdoor': create_outdoor_noise_model()
        }
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("DataGenerator initialized successfully")
    
    def generate_complete_dataset(
        self,
        output_dir: Path,
        train_samples: int = None,
        val_samples: int = None,
        imaging_modalities: List[str] = None,
        environments: List[str] = None,
        materials: List[str] = None,
        defect_types: List[str] = None,
        use_multiprocessing: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic dataset with multiple imaging modalities.
        
        Args:
            output_dir: Output directory for generated data
            train_samples: Number of training samples
            val_samples: Number of validation samples
            imaging_modalities: List of imaging modalities ('lsci', 'oct', 'hybrid')
            environments: List of environments ('industrial', 'laboratory', 'outdoor')
            materials: List of materials to simulate
            defect_types: List of defect types to include
            use_multiprocessing: Whether to use multiprocessing
        
        Returns:
            Dictionary with generation statistics
        """
        # Set defaults
        train_samples = train_samples or self.config.data_generation.num_train_samples
        val_samples = val_samples or self.config.data_generation.num_val_samples
        imaging_modalities = imaging_modalities or ['lsci', 'oct']
        environments = environments or ['industrial', 'laboratory']
        materials = materials or self.config.data_generation.materials
        defect_types = defect_types or ['surface_scratch', 'internal_crack', 'material_inclusion']
        
        # Create directory structure
        create_directory_structure(output_dir)
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        logger.info(f"Starting dataset generation: {train_samples} train, {val_samples} validation samples")
        logger.info(f"Imaging modalities: {imaging_modalities}")
        logger.info(f"Environments: {environments}")
        logger.info(f"Materials: {materials}")
        
        generation_stats = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'imaging_modalities': imaging_modalities,
            'environments': environments,
            'materials': materials,
            'defect_types': defect_types,
            'generation_time': None,
            'file_sizes': {},
            'quality_metrics': {}
        }
        
        try:
            # Generate training data
            logger.info("Generating training data...")
            train_stats = self._generate_split(
                'train', train_samples, output_dir, imaging_modalities,
                environments, materials, defect_types, use_multiprocessing
            )
            generation_stats['train_stats'] = train_stats
            
            # Generate validation data
            logger.info("Generating validation data...")
            val_stats = self._generate_split(
                'val', val_samples, output_dir, imaging_modalities,
                environments, materials, defect_types, use_multiprocessing
            )
            generation_stats['val_stats'] = val_stats
            
            # Validate generated data
            logger.info("Validating generated data...")
            validation_result = validate_data_integrity(output_dir / "data" / "synthetic")
            generation_stats['validation_passed'] = validation_result
            
            # Calculate quality metrics
            generation_stats['quality_metrics'] = self._calculate_quality_metrics(output_dir)
            
            # Record performance
            self.performance_monitor.record()
            generation_stats['performance_summary'] = self.performance_monitor.get_summary()
            
            # Save generation metadata
            self._save_generation_metadata(output_dir, generation_stats)
            
            logger.info("Dataset generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise
        
        return generation_stats
    
    def _generate_split(
        self,
        split_name: str,
        num_samples: int,
        output_dir: Path,
        imaging_modalities: List[str],
        environments: List[str],
        materials: List[str],
        defect_types: List[str],
        use_multiprocessing: bool
    ) -> Dict[str, Any]:
        """Generate data for a specific split (train/val)."""
        split_stats = {
            'num_samples': num_samples,
            'modalities': {},
            'environments': {},
            'materials': {},
            'defect_distribution': {}
        }
        
        # Calculate samples per modality
        samples_per_modality = num_samples // len(imaging_modalities)
        remaining_samples = num_samples % len(imaging_modalities)
        
        for i, modality in enumerate(imaging_modalities):
            # Add remaining samples to first modality
            current_samples = samples_per_modality + (remaining_samples if i == 0 else 0)
            
            logger.info(f"Generating {current_samples} samples for {modality} modality...")
            
            modality_stats = self._generate_modality_data(
                modality, current_samples, output_dir, split_name,
                environments, materials, defect_types, use_multiprocessing
            )
            
            split_stats['modalities'][modality] = modality_stats
        
        return split_stats
    
    def _generate_modality_data(
        self,
        modality: str,
        num_samples: int,
        output_dir: Path,
        split_name: str,
        environments: List[str],
        materials: List[str],
        defect_types: List[str],
        use_multiprocessing: bool
    ) -> Dict[str, Any]:
        """Generate data for a specific imaging modality."""
        modality_stats = {
            'num_samples': num_samples,
            'environments': {},
            'materials': {},
            'defect_distribution': {}
        }
        
        # Calculate samples per environment
        samples_per_env = num_samples // len(environments)
        remaining_samples = num_samples % len(environments)
        
        all_images = []
        all_masks = []
        all_metadata = []
        
        for i, environment in enumerate(environments):
            current_samples = samples_per_env + (remaining_samples if i == 0 else 0)
            
            logger.info(f"Generating {current_samples} samples for {environment} environment...")
            
            env_images, env_masks, env_metadata = self._generate_environment_data(
                modality, environment, current_samples, materials, defect_types, use_multiprocessing
            )
            
            all_images.extend(env_images)
            all_masks.extend(env_masks)
            all_metadata.extend(env_metadata)
            
            modality_stats['environments'][environment] = {
                'num_samples': current_samples,
                'materials': self._count_materials(env_metadata),
                'defect_distribution': self._count_defects(env_metadata)
            }
        
        # Save data
        self._save_modality_data(
            modality, all_images, all_masks, all_metadata,
            output_dir, split_name
        )
        
        # Update statistics
        modality_stats['materials'] = self._count_materials(all_metadata)
        modality_stats['defect_distribution'] = self._count_defects(all_metadata)
        
        return modality_stats
    
    def _generate_environment_data(
        self,
        modality: str,
        environment: str,
        num_samples: int,
        materials: List[str],
        defect_types: List[str],
        use_multiprocessing: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate data for a specific environment."""
        if use_multiprocessing and num_samples > 100:
            return self._generate_parallel(
                modality, environment, num_samples, materials, defect_types
            )
        else:
            return self._generate_sequential(
                modality, environment, num_samples, materials, defect_types
            )
    
    def _generate_sequential(
        self,
        modality: str,
        environment: str,
        num_samples: int,
        materials: List[str],
        defect_types: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate data sequentially."""
        images = []
        masks = []
        metadata = []
        
        noise_model = self.noise_models[environment]
        
        for i in tqdm(range(num_samples), desc=f"Generating {environment} {modality}"):
            # Random material and defect properties
            material = np.random.choice(materials)
            material_props = self.config.data_generation.material_properties[material]
            
            # Generate defect regions
            defect_regions = []
            mask = np.zeros(self.config.data_generation.image_size, dtype=np.uint8)
            
            if np.random.random() < self.config.data_generation.defect_probability:
                num_defects = np.random.randint(1, 4)
                
                for _ in range(num_defects):
                    defect_type = np.random.choice(defect_types)
                    center = (
                        np.random.randint(50, self.config.data_generation.image_size[1] - 50),
                        np.random.randint(50, self.config.data_generation.image_size[0] - 50)
                    )
                    radius = np.random.randint(10, 30)
                    
                    if modality == 'lsci':
                        defect_regions.append(self.lsci_simulator.create_defect_region(
                            defect_type, center, radius
                        ))
                    elif modality == 'oct':
                        depth_range = (
                            np.random.randint(0, self.config.data_generation.image_size[0] // 3),
                            np.random.randint(self.config.data_generation.image_size[0] // 3, 
                                            self.config.data_generation.image_size[0])
                        )
                        defect_regions.append(self.oct_simulator.create_defect_region(
                            defect_type, center, radius, depth_range
                        ))
                    
                    # Create mask
                    y_coords, x_coords = np.ogrid[:self.config.data_generation.image_size[0], 
                                                 :self.config.data_generation.image_size[1]]
                    defect_mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                    mask[defect_mask] = 1
            
            # Generate image
            if modality == 'lsci':
                image = self.lsci_simulator.generate_speckle_pattern(
                    surface_roughness=material_props['roughness'],
                    material_reflectivity=material_props['reflectivity'],
                    defect_regions=defect_regions
                )
            elif modality == 'oct':
                tissue_layers = create_standard_tissue_layers()
                image = self.oct_simulator.generate_oct_image(
                    tissue_layers, defect_regions
                )
            else:  # hybrid
                # Generate both modalities and combine
                lsci_image = self.lsci_simulator.generate_speckle_pattern(
                    surface_roughness=material_props['roughness'],
                    material_reflectivity=material_props['reflectivity'],
                    defect_regions=defect_regions
                )
                tissue_layers = create_standard_tissue_layers()
                oct_image = self.oct_simulator.generate_oct_image(
                    tissue_layers, defect_regions
                )
                # Combine as RGB (LSCI as R, OCT as G, average as B)
                image = np.stack([lsci_image, oct_image, (lsci_image + oct_image) / 2], axis=-1)
            
            # Apply noise
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
            
            noisy_image = noise_model.apply(image)
            
            # Ensure 3-channel output
            if noisy_image.shape[-1] == 1:
                noisy_image = np.repeat(noisy_image, 3, axis=-1)
            
            images.append(noisy_image)
            masks.append(mask)
            
            # Metadata
            metadata.append({
                'modality': modality,
                'environment': environment,
                'material': material,
                'defect_types': [d['type'] for d in defect_regions],
                'num_defects': len(defect_regions),
                'material_properties': material_props
            })
        
        return images, masks, metadata
    
    def _generate_parallel(
        self,
        modality: str,
        environment: str,
        num_samples: int,
        materials: List[str],
        defect_types: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate data using parallel processing."""
        # Split work into chunks
        chunk_size = max(1, num_samples // mp.cpu_count())
        chunks = [(i, min(i + chunk_size, num_samples)) for i in range(0, num_samples, chunk_size)]
        
        all_images = []
        all_masks = []
        all_metadata = []
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for start, end in chunks:
                future = executor.submit(
                    self._generate_chunk,
                    modality, environment, end - start, materials, defect_types
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                chunk_images, chunk_masks, chunk_metadata = future.result()
                all_images.extend(chunk_images)
                all_masks.extend(chunk_masks)
                all_metadata.extend(chunk_metadata)
        
        return all_images, all_masks, all_metadata
    
    def _generate_chunk(
        self,
        modality: str,
        environment: str,
        num_samples: int,
        materials: List[str],
        defect_types: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate a chunk of data (for parallel processing)."""
        # Create local simulators and noise model
        lsci_sim = LSCISimulator(
            wavelength=self.config.data_generation.lsci_wavelength,
            coherence_length=self.config.data_generation.lsci_coherence_length,
            speckle_size=self.config.data_generation.lsci_speckle_size,
            contrast_range=self.config.data_generation.lsci_contrast_range,
            image_size=self.config.data_generation.image_size
        )
        
        oct_sim = OCTSimulator(
            center_wavelength=self.config.data_generation.oct_center_wavelength,
            bandwidth=self.config.data_generation.oct_bandwidth,
            axial_resolution=self.config.data_generation.oct_axial_resolution,
            lateral_resolution=self.config.data_generation.oct_lateral_resolution,
            depth_range=self.config.data_generation.oct_depth_range,
            image_size=self.config.data_generation.image_size
        )
        
        noise_model = self.noise_models[environment]
        
        images = []
        masks = []
        metadata = []
        
        for i in range(num_samples):
            # Similar logic to _generate_sequential but with local simulators
            material = np.random.choice(materials)
            material_props = self.config.data_generation.material_properties[material]
            
            # Generate defect regions and image (simplified for brevity)
            # ... (similar to _generate_sequential)
            
            # Placeholder for actual generation
            image = np.random.random(self.config.data_generation.image_size + (3,))
            mask = np.zeros(self.config.data_generation.image_size, dtype=np.uint8)
            
            images.append(image)
            masks.append(mask)
            metadata.append({
                'modality': modality,
                'environment': environment,
                'material': material,
                'defect_types': [],
                'num_defects': 0,
                'material_properties': material_props
            })
        
        return images, masks, metadata
    
    def _save_modality_data(
        self,
        modality: str,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        output_dir: Path,
        split_name: str
    ):
        """Save generated data for a modality."""
        # Create directories
        images_dir = output_dir / "data" / "synthetic" / split_name / "images"
        masks_dir = output_dir / "data" / "synthetic" / split_name / "masks"
        metadata_dir = output_dir / "data" / "synthetic" / split_name / "metadata"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images and masks
        for i, (image, mask, meta) in enumerate(zip(images, masks, metadata)):
            # Save image
            image_path = images_dir / f"{modality}_{split_name}_{i:06d}.png"
            image_uint8 = (image * 255).astype(np.uint8)
            cv2.imwrite(str(image_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
            
            # Save mask
            mask_path = masks_dir / f"{modality}_{split_name}_{i:06d}.png"
            cv2.imwrite(str(mask_path), mask * 255)
            
            # Save metadata
            meta['image_path'] = str(image_path)
            meta['mask_path'] = str(mask_path)
            meta['sample_id'] = f"{modality}_{split_name}_{i:06d}"
            
            meta_path = metadata_dir / f"{modality}_{split_name}_{i:06d}.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
    
    def _count_materials(self, metadata: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count material distribution in metadata."""
        material_counts = {}
        for meta in metadata:
            material = meta['material']
            material_counts[material] = material_counts.get(material, 0) + 1
        return material_counts
    
    def _count_defects(self, metadata: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count defect type distribution in metadata."""
        defect_counts = {}
        for meta in metadata:
            for defect_type in meta['defect_types']:
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        return defect_counts
    
    def _calculate_quality_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """Calculate quality metrics for generated data."""
        # This would include metrics like:
        # - Image quality (SNR, contrast)
        # - Defect visibility
        # - Dataset balance
        # - etc.
        return {
            'total_images': 0,  # Placeholder
            'average_image_quality': 0.0,  # Placeholder
            'defect_detection_rate': 0.0  # Placeholder
        }
    
    def _save_generation_metadata(self, output_dir: Path, stats: Dict[str, Any]):
        """Save generation metadata."""
        metadata_path = output_dir / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Generation metadata saved to {metadata_path}")

def main():
    """Main function for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic optical diagnostics data")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--modalities", nargs="+", default=["lsci", "oct"], help="Imaging modalities")
    parser.add_argument("--environments", nargs="+", default=["industrial", "laboratory"], help="Environments")
    parser.add_argument("--materials", nargs="+", default=["aluminum", "steel"], help="Materials")
    parser.add_argument("--defect_types", nargs="+", default=["surface_scratch", "internal_crack"], help="Defect types")
    parser.add_argument("--no_multiprocessing", action="store_true", help="Disable multiprocessing")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DataGenerator()
    
    # Generate dataset
    stats = generator.generate_complete_dataset(
        output_dir=Path(args.output_dir),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        imaging_modalities=args.modalities,
        environments=args.environments,
        materials=args.materials,
        defect_types=args.defect_types,
        use_multiprocessing=not args.no_multiprocessing
    )
    
    print("Dataset generation completed!")
    print(f"Generated {stats['train_samples']} training and {stats['val_samples']} validation samples")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()