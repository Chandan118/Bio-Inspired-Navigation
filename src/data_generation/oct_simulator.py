"""
Optical Coherence Tomography (OCT) simulator.
Implements physics-based simulation of OCT imaging with realistic tissue and defect modeling.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import gamma, norm, rayleigh
from scipy.signal import hilbert, fftconvolve
from typing import Tuple, Optional, Dict, Any, List
import logging
from numba import jit, prange
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class OCTSimulator:
    """
    Optical Coherence Tomography simulator.
    
    Simulates OCT imaging based on:
    - Light source properties (wavelength, bandwidth)
    - Tissue optical properties (scattering, absorption)
    - Axial and lateral resolution
    - Depth-dependent signal attenuation
    - Defect scattering characteristics
    """
    
    def __init__(
        self,
        center_wavelength: float = 1300.0,  # nm
        bandwidth: float = 100.0,  # nm
        axial_resolution: float = 5.0,  # μm
        lateral_resolution: float = 10.0,  # μm
        depth_range: float = 2000.0,  # μm
        image_size: Tuple[int, int] = (512, 512),
        pixel_size: float = 5.0  # μm
    ):
        self.center_wavelength = center_wavelength * 1e-9  # Convert to meters
        self.bandwidth = bandwidth * 1e-9  # Convert to meters
        self.axial_resolution = axial_resolution * 1e-6  # Convert to meters
        self.lateral_resolution = lateral_resolution * 1e-6  # Convert to meters
        self.depth_range = depth_range * 1e-6  # Convert to meters
        self.image_size = image_size
        self.pixel_size = pixel_size * 1e-6  # Convert to meters
        
        # Calculate derived parameters
        self.coherence_length = (self.center_wavelength**2) / (2 * self.bandwidth)
        self.axial_pixels = int(self.depth_range / self.axial_resolution)
        self.lateral_pixels = int(self.image_size[1] / (self.lateral_resolution / self.pixel_size))
        
        logger.info(f"OCT Simulator initialized with axial resolution: {self.axial_resolution*1e6:.2f} μm")
    
    def generate_oct_image(
        self,
        tissue_layers: List[Dict[str, Any]],
        defect_regions: Optional[List[Dict[str, Any]]] = None,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Generate a realistic OCT image.
        
        Args:
            tissue_layers: List of tissue layer specifications
            defect_regions: List of defect regions
            noise_level: Noise level (0-1)
        
        Returns:
            OCT image (depth x lateral)
        """
        # Initialize OCT image
        oct_image = np.zeros((self.axial_pixels, self.image_size[1]))
        
        # Generate tissue structure
        tissue_structure = self._generate_tissue_structure(tissue_layers)
        
        # Apply defect modifications
        if defect_regions:
            tissue_structure = self._apply_defect_modifications(tissue_structure, defect_regions)
        
        # Generate OCT signal
        oct_signal = self._generate_oct_signal(tissue_structure)
        
        # Apply depth-dependent attenuation
        oct_signal = self._apply_depth_attenuation(oct_signal)
        
        # Apply lateral resolution effects
        oct_signal = self._apply_lateral_resolution(oct_signal)
        
        # Add noise
        oct_signal = self._add_oct_noise(oct_signal, noise_level)
        
        # Convert to logarithmic scale (typical for OCT display)
        oct_image = 20 * np.log10(oct_signal + 1e-10)
        
        # Normalize to [0, 1]
        oct_image = (oct_image - np.min(oct_image)) / (np.max(oct_image) - np.min(oct_image))
        
        return oct_image
    
    def _generate_tissue_structure(self, tissue_layers: List[Dict[str, Any]]) -> np.ndarray:
        """Generate realistic tissue structure with multiple layers."""
        tissue_structure = np.zeros((self.axial_pixels, self.image_size[1]))
        
        current_depth = 0
        
        for layer in tissue_layers:
            layer_thickness = layer['thickness'] * 1e-6  # Convert to meters
            layer_pixels = int(layer_thickness / self.axial_resolution)
            
            # Generate layer properties
            scattering_coeff = layer['scattering_coefficient']
            absorption_coeff = layer['absorption_coefficient']
            refractive_index = layer['refractive_index']
            
            # Create layer with realistic variations
            layer_structure = self._create_layer_structure(
                layer_pixels, self.image_size[1],
                scattering_coeff, absorption_coeff, refractive_index
            )
            
            # Add to tissue structure
            end_depth = min(current_depth + layer_pixels, self.axial_pixels)
            tissue_structure[current_depth:end_depth, :] = layer_structure[:end_depth-current_depth, :]
            
            current_depth = end_depth
            
            if current_depth >= self.axial_pixels:
                break
        
        return tissue_structure
    
    def _create_layer_structure(
        self,
        depth_pixels: int,
        width_pixels: int,
        scattering_coeff: float,
        absorption_coeff: float,
        refractive_index: float
    ) -> np.ndarray:
        """Create realistic layer structure with spatial variations."""
        layer = np.zeros((depth_pixels, width_pixels))
        
        # Base scattering coefficient
        base_scattering = scattering_coeff * np.ones((depth_pixels, width_pixels))
        
        # Add spatial variations (fractal-like structure)
        variations = self._generate_spatial_variations(depth_pixels, width_pixels)
        base_scattering *= (1 + 0.3 * variations)
        
        # Add depth-dependent scattering (increases with depth due to tissue complexity)
        depth_factor = np.linspace(1.0, 1.5, depth_pixels)[:, np.newaxis]
        base_scattering *= depth_factor
        
        # Add absorption effects
        absorption_factor = np.exp(-absorption_coeff * np.arange(depth_pixels)[:, np.newaxis] * self.axial_resolution)
        base_scattering *= absorption_factor
        
        return base_scattering
    
    def _generate_spatial_variations(self, height: int, width: int) -> np.ndarray:
        """Generate fractal-like spatial variations."""
        # Create frequency grid
        fx = np.fft.fftfreq(width)
        fy = np.fft.fftfreq(height)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Power law spectrum (fractal-like)
        freq_magnitude = np.sqrt(fx_grid**2 + fy_grid**2)
        freq_magnitude[0, 0] = 1e-10  # Avoid division by zero
        
        # Power law with exponent -2 (typical for tissue)
        power_spectrum = freq_magnitude ** (-2)
        
        # Generate random phase
        random_phase = np.random.uniform(0, 2 * np.pi, (height, width))
        
        # Create complex spectrum
        complex_spectrum = np.sqrt(power_spectrum) * np.exp(1j * random_phase)
        
        # Inverse FFT to get spatial variations
        variations = np.real(np.fft.ifft2(complex_spectrum))
        
        # Normalize
        variations = (variations - np.mean(variations)) / np.std(variations)
        
        return variations
    
    def _apply_defect_modifications(
        self,
        tissue_structure: np.ndarray,
        defect_regions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Apply defect modifications to tissue structure."""
        modified_structure = tissue_structure.copy()
        
        for defect in defect_regions:
            x, y = defect['center']
            radius = defect['radius']
            defect_type = defect['type']
            
            # Create defect mask
            y_coords, x_coords = np.ogrid[:tissue_structure.shape[0], :tissue_structure.shape[1]]
            mask = (x_coords - y)**2 + (y_coords - x)**2 <= radius**2
            
            if defect_type == 'internal_crack':
                # Internal crack: reduced scattering
                modified_structure[mask] *= 0.2
                
            elif defect_type == 'material_inclusion':
                # Material inclusion: increased scattering
                modified_structure[mask] *= 2.0
                
            elif defect_type == 'boundary_defect':
                # Boundary defect: interface scattering
                modified_structure[mask] *= 1.5
                
            elif defect_type == 'thermal_damage':
                # Thermal damage: increased absorption
                modified_structure[mask] *= 0.5
                
            elif defect_type == 'stress_concentration':
                # Stress concentration: anisotropic scattering
                # Create directional scattering pattern
                angle = np.arctan2(y_coords - y, x_coords - x)
                directional_factor = 1 + 0.5 * np.cos(2 * angle)
                modified_structure[mask] *= directional_factor[mask]
        
        return modified_structure
    
    def _generate_oct_signal(self, tissue_structure: np.ndarray) -> np.ndarray:
        """Generate OCT signal from tissue structure."""
        # OCT signal is proportional to backscattered intensity
        # Use Rayleigh scattering model
        backscattered_intensity = tissue_structure ** 2
        
        # Add speckle noise (characteristic of OCT)
        speckle_factor = np.random.rayleigh(1.0, backscattered_intensity.shape)
        oct_signal = backscattered_intensity * speckle_factor
        
        # Apply coherence gating (axial resolution)
        coherence_kernel = self._create_coherence_kernel()
        oct_signal = fftconvolve(oct_signal, coherence_kernel[np.newaxis, :], mode='same', axes=1)
        
        return oct_signal
    
    def _create_coherence_kernel(self) -> np.ndarray:
        """Create coherence kernel based on source properties."""
        # Gaussian envelope with coherence length
        kernel_size = max(3, int(self.coherence_length / self.axial_resolution))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        center = kernel_size // 2
        kernel = np.zeros(kernel_size)
        
        for i in range(kernel_size):
            distance = abs(i - center) * self.axial_resolution
            kernel[i] = np.exp(-(distance / self.coherence_length) ** 2)
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _apply_depth_attenuation(self, oct_signal: np.ndarray) -> np.ndarray:
        """Apply depth-dependent signal attenuation."""
        depth_pixels = oct_signal.shape[0]
        
        # Exponential attenuation with depth
        depth_coords = np.arange(depth_pixels)[:, np.newaxis] * self.axial_resolution
        attenuation_factor = np.exp(-depth_coords / (self.depth_range * 0.1))  # 10% of depth range
        
        # Apply attenuation
        attenuated_signal = oct_signal * attenuation_factor
        
        return attenuated_signal
    
    def _apply_lateral_resolution(self, oct_signal: np.ndarray) -> np.ndarray:
        """Apply lateral resolution effects."""
        # Create lateral resolution kernel
        kernel_size = max(3, int(self.lateral_resolution / self.pixel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = self.lateral_resolution / (2 * self.pixel_size)
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        
        # Apply lateral filtering
        filtered_signal = cv2.filter2D(oct_signal, -1, kernel)
        
        return filtered_signal
    
    def _add_oct_noise(self, oct_signal: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic OCT noise."""
        # Shot noise (Poisson)
        shot_noise = np.random.poisson(oct_signal * 1000) / 1000
        
        # Thermal noise (Gaussian)
        thermal_noise = np.random.normal(0, noise_level * np.mean(oct_signal), oct_signal.shape)
        
        # Electronic noise
        electronic_noise = np.random.normal(0, noise_level * 0.1 * np.mean(oct_signal), oct_signal.shape)
        
        # Combine noises
        noisy_signal = oct_signal + shot_noise + thermal_noise + electronic_noise
        
        # Ensure non-negative
        noisy_signal = np.maximum(noisy_signal, 0)
        
        return noisy_signal
    
    def create_tissue_layer(
        self,
        thickness: float,  # μm
        scattering_coefficient: float = 100.0,  # cm^-1
        absorption_coefficient: float = 1.0,  # cm^-1
        refractive_index: float = 1.4
    ) -> Dict[str, Any]:
        """Create a tissue layer specification."""
        return {
            'thickness': thickness,
            'scattering_coefficient': scattering_coefficient,
            'absorption_coefficient': absorption_coefficient,
            'refractive_index': refractive_index
        }
    
    def create_defect_region(
        self,
        defect_type: str,
        center: Tuple[int, int],
        radius: int,
        depth_range: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Create a defect region specification."""
        return {
            'type': defect_type,
            'center': center,
            'radius': radius,
            'depth_range': depth_range
        }
    
    def generate_b_scan(
        self,
        tissue_layers: List[Dict[str, Any]],
        defect_regions: Optional[List[Dict[str, Any]]] = None,
        num_a_scans: int = 512
    ) -> np.ndarray:
        """
        Generate a B-scan (2D cross-sectional image).
        
        Args:
            tissue_layers: Tissue layer specifications
            defect_regions: Defect regions
            num_a_scans: Number of A-scans (lateral positions)
        
        Returns:
            B-scan image
        """
        # Update image size for B-scan
        original_size = self.image_size
        self.image_size = (self.axial_pixels, num_a_scans)
        
        # Generate B-scan
        b_scan = self.generate_oct_image(tissue_layers, defect_regions)
        
        # Restore original image size
        self.image_size = original_size
        
        return b_scan
    
    def generate_3d_volume(
        self,
        tissue_layers: List[Dict[str, Any]],
        defect_regions: Optional[List[Dict[str, Any]]] = None,
        num_b_scans: int = 100
    ) -> np.ndarray:
        """
        Generate a 3D OCT volume.
        
        Args:
            tissue_layers: Tissue layer specifications
            defect_regions: Defect regions
            num_b_scans: Number of B-scans
        
        Returns:
            3D OCT volume
        """
        volume = []
        
        for i in range(num_b_scans):
            # Modify defect positions for 3D effect
            modified_defects = self._modify_defects_for_3d(defect_regions, i, num_b_scans)
            
            # Generate B-scan
            b_scan = self.generate_b_scan(tissue_layers, modified_defects)
            volume.append(b_scan)
        
        return np.array(volume)
    
    def _modify_defects_for_3d(
        self,
        defect_regions: Optional[List[Dict[str, Any]]],
        scan_index: int,
        total_scans: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Modify defect positions for 3D volume generation."""
        if defect_regions is None:
            return None
        
        modified_defects = []
        
        for defect in defect_regions:
            # Add slight variations in position for 3D effect
            x, y = defect['center']
            x_variation = np.random.normal(0, 2)  # Small lateral variation
            y_variation = np.random.normal(0, 1)  # Small depth variation
            
            modified_defect = defect.copy()
            modified_defect['center'] = (x + x_variation, y + y_variation)
            modified_defects.append(modified_defect)
        
        return modified_defects
    
    def visualize_oct_analysis(
        self,
        oct_image: np.ndarray,
        tissue_layers: Optional[List[Dict[str, Any]]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize OCT image and analysis.
        
        Args:
            oct_image: OCT image
            tissue_layers: Tissue layer specifications
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # OCT image
        im1 = axes[0].imshow(oct_image, cmap='gray', aspect='auto')
        axes[0].set_title('OCT B-Scan')
        axes[0].set_xlabel('Lateral Position')
        axes[0].set_ylabel('Depth')
        plt.colorbar(im1, ax=axes[0])
        
        # Depth profile
        depth_profile = np.mean(oct_image, axis=1)
        axes[1].plot(depth_profile, np.arange(len(depth_profile)))
        axes[1].set_title('Average Depth Profile')
        axes[1].set_xlabel('Signal Intensity')
        axes[1].set_ylabel('Depth')
        axes[1].invert_yaxis()
        
        # Add tissue layer boundaries if provided
        if tissue_layers:
            current_depth = 0
            for layer in tissue_layers:
                layer_pixels = int(layer['thickness'] * 1e-6 / self.axial_resolution)
                current_depth += layer_pixels
                if current_depth < len(depth_profile):
                    axes[1].axhline(y=current_depth, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"OCT analysis visualization saved to {save_path}")
        
        plt.show()

def create_standard_tissue_layers() -> List[Dict[str, Any]]:
    """Create standard tissue layer configuration."""
    layers = [
        {
            'thickness': 100,  # μm
            'scattering_coefficient': 50.0,  # cm^-1
            'absorption_coefficient': 0.5,  # cm^-1
            'refractive_index': 1.33
        },
        {
            'thickness': 200,  # μm
            'scattering_coefficient': 150.0,  # cm^-1
            'absorption_coefficient': 2.0,  # cm^-1
            'refractive_index': 1.4
        },
        {
            'thickness': 300,  # μm
            'scattering_coefficient': 200.0,  # cm^-1
            'absorption_coefficient': 5.0,  # cm^-1
            'refractive_index': 1.45
        }
    ]
    return layers

def create_oct_dataset(
    num_samples: int,
    image_size: Tuple[int, int] = (512, 512),
    defect_probability: float = 0.3,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of OCT images with corresponding defect masks.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of generated images
        defect_probability: Probability of including defects
        **kwargs: Additional arguments for OCT simulator
    
    Returns:
        Tuple of (images, masks)
    """
    simulator = OCTSimulator(image_size=image_size, **kwargs)
    
    images = []
    masks = []
    
    for i in range(num_samples):
        # Generate tissue layers
        tissue_layers = create_standard_tissue_layers()
        
        # Generate defect regions
        defect_regions = []
        mask = np.zeros(image_size, dtype=np.uint8)
        
        if np.random.random() < defect_probability:
            num_defects = np.random.randint(1, 3)
            
            for _ in range(num_defects):
                # Random defect properties
                defect_type = np.random.choice([
                    'internal_crack', 'material_inclusion', 'boundary_defect',
                    'thermal_damage', 'stress_concentration'
                ])
                
                center = (
                    np.random.randint(50, image_size[1] - 50),
                    np.random.randint(50, image_size[0] - 50)
                )
                radius = np.random.randint(5, 20)
                depth_range = (
                    np.random.randint(0, image_size[0] // 3),
                    np.random.randint(image_size[0] // 3, image_size[0])
                )
                
                defect_regions.append(simulator.create_defect_region(
                    defect_type, center, radius, depth_range
                ))
                
                # Create mask
                y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
                defect_mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                mask[defect_mask] = 1
        
        # Generate OCT image
        oct_image = simulator.generate_oct_image(tissue_layers, defect_regions)
        
        images.append(oct_image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)