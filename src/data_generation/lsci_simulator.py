"""
Laser Speckle Contrast Imaging (LSCI) simulator.
Implements physics-based simulation of laser speckle patterns and contrast analysis.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import gamma, poisson
from scipy.signal import convolve2d
from typing import Tuple, Optional, Dict, Any, List
import logging
from numba import jit, prange
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class LSCISimulator:
    """
    Laser Speckle Contrast Imaging simulator.
    
    Simulates laser speckle patterns based on:
    - Laser wavelength and coherence properties
    - Surface roughness and material properties
    - Speckle size and contrast relationships
    - Temporal and spatial statistics
    """
    
    def __init__(
        self,
        wavelength: float = 785.0,  # nm
        coherence_length: float = 50.0,  # μm
        speckle_size: float = 2.0,  # μm
        contrast_range: Tuple[float, float] = (0.1, 0.8),
        image_size: Tuple[int, int] = (512, 512),
        pixel_size: float = 5.0  # μm
    ):
        self.wavelength = wavelength * 1e-9  # Convert to meters
        self.coherence_length = coherence_length * 1e-6  # Convert to meters
        self.speckle_size = speckle_size * 1e-6  # Convert to meters
        self.contrast_range = contrast_range
        self.image_size = image_size
        self.pixel_size = pixel_size * 1e-6  # Convert to meters
        
        # Calculate derived parameters
        self.speckle_pixels = int(self.speckle_size / self.pixel_size)
        self.coherence_pixels = int(self.coherence_length / self.pixel_size)
        
        logger.info(f"LSCI Simulator initialized with speckle size: {self.speckle_size*1e6:.2f} μm")
    
    def generate_speckle_pattern(
        self,
        surface_roughness: float = 0.1,
        material_reflectivity: float = 0.8,
        defect_regions: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Generate a realistic laser speckle pattern.
        
        Args:
            surface_roughness: RMS surface roughness (μm)
            material_reflectivity: Material reflectivity (0-1)
            defect_regions: List of defect regions with properties
        
        Returns:
            Speckle pattern image
        """
        height, width = self.image_size
        
        # Generate complex field
        complex_field = self._generate_complex_field(surface_roughness, material_reflectivity)
        
        # Apply defect modifications
        if defect_regions:
            complex_field = self._apply_defect_modifications(complex_field, defect_regions)
        
        # Calculate intensity pattern
        intensity_pattern = np.abs(complex_field) ** 2
        
        # Apply speckle statistics
        speckle_pattern = self._apply_speckle_statistics(intensity_pattern)
        
        # Normalize to [0, 1]
        speckle_pattern = (speckle_pattern - np.min(speckle_pattern)) / (np.max(speckle_pattern) - np.min(speckle_pattern))
        
        return speckle_pattern
    
    def _generate_complex_field(
        self,
        surface_roughness: float,
        material_reflectivity: float
    ) -> np.ndarray:
        """Generate complex optical field based on surface properties."""
        height, width = self.image_size
        
        # Surface height profile (Gaussian random field)
        surface_height = self._generate_surface_height(surface_roughness)
        
        # Phase variation due to surface height
        phase_variation = 2 * np.pi * surface_height / self.wavelength
        
        # Amplitude variation due to material properties
        amplitude_variation = np.sqrt(material_reflectivity) * np.ones_like(phase_variation)
        
        # Add spatial correlation based on coherence length
        correlation_kernel = self._create_correlation_kernel()
        amplitude_variation = convolve2d(amplitude_variation, correlation_kernel, mode='same')
        
        # Complex field
        complex_field = amplitude_variation * np.exp(1j * phase_variation)
        
        return complex_field
    
    def _generate_surface_height(self, roughness: float) -> np.ndarray:
        """Generate surface height profile using power spectral density."""
        height, width = self.image_size
        
        # Create frequency grid
        fx = np.fft.fftfreq(width)
        fy = np.fft.fftfreq(height)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Power spectral density (PSD) for rough surfaces
        # Using modified power law PSD
        freq_magnitude = np.sqrt(fx_grid**2 + fy_grid**2)
        freq_magnitude[0, 0] = 1e-10  # Avoid division by zero
        
        # Power law PSD with cutoff
        cutoff_freq = 1.0 / (self.speckle_size / self.pixel_size)
        psd = roughness**2 / (1 + (freq_magnitude / cutoff_freq)**2)
        
        # Generate random phase
        random_phase = np.random.uniform(0, 2 * np.pi, (height, width))
        
        # Create complex spectrum
        complex_spectrum = np.sqrt(psd) * np.exp(1j * random_phase)
        
        # Inverse FFT to get surface height
        surface_height = np.real(np.fft.ifft2(complex_spectrum))
        
        return surface_height
    
    def _create_correlation_kernel(self) -> np.ndarray:
        """Create spatial correlation kernel based on coherence length."""
        kernel_size = max(3, int(self.coherence_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size))
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-distance / (self.coherence_pixels / 2))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _apply_defect_modifications(
        self,
        complex_field: np.ndarray,
        defect_regions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Apply defect modifications to the complex field."""
        modified_field = complex_field.copy()
        
        for defect in defect_regions:
            x, y = defect['center']
            radius = defect['radius']
            defect_type = defect['type']
            
            # Create defect mask
            y_coords, x_coords = np.ogrid[:self.image_size[0], :self.image_size[1]]
            mask = (x_coords - x)**2 + (y_coords - y)**2 <= radius**2
            
            if defect_type == 'surface_scratch':
                # Surface scratch: reduced reflectivity and increased roughness
                modified_field[mask] *= 0.3 * np.exp(1j * np.random.uniform(0, 2*np.pi, np.sum(mask)))
            
            elif defect_type == 'internal_crack':
                # Internal crack: phase discontinuity
                phase_jump = np.pi * np.random.uniform(0.5, 1.5)
                modified_field[mask] *= np.exp(1j * phase_jump)
            
            elif defect_type == 'material_inclusion':
                # Material inclusion: different optical properties
                modified_field[mask] *= 0.7 * np.exp(1j * np.random.uniform(-np.pi/4, np.pi/4, np.sum(mask)))
            
            elif defect_type == 'thermal_damage':
                # Thermal damage: increased surface roughness
                roughness_factor = np.random.uniform(2.0, 5.0)
                phase_variation = roughness_factor * np.random.uniform(-np.pi, np.pi, np.sum(mask))
                modified_field[mask] *= np.exp(1j * phase_variation)
        
        return modified_field
    
    def _apply_speckle_statistics(self, intensity_pattern: np.ndarray) -> np.ndarray:
        """Apply realistic speckle statistics."""
        # Speckle contrast calculation
        speckle_contrast = self._calculate_speckle_contrast(intensity_pattern)
        
        # Apply gamma distribution for speckle intensity
        # Gamma distribution is characteristic of speckle patterns
        shape_param = 1.0 / speckle_contrast**2
        scale_param = intensity_pattern / shape_param
        
        # Generate gamma-distributed speckle pattern
        speckle_pattern = np.random.gamma(shape_param, scale_param)
        
        # Apply spatial filtering to match speckle size
        speckle_pattern = self._apply_spatial_filtering(speckle_pattern)
        
        return speckle_pattern
    
    def _calculate_speckle_contrast(self, intensity_pattern: np.ndarray) -> float:
        """Calculate speckle contrast from intensity pattern."""
        mean_intensity = np.mean(intensity_pattern)
        std_intensity = np.std(intensity_pattern)
        
        if mean_intensity > 0:
            contrast = std_intensity / mean_intensity
        else:
            contrast = 0.1
        
        # Ensure contrast is within reasonable bounds
        contrast = np.clip(contrast, self.contrast_range[0], self.contrast_range[1])
        
        return contrast
    
    def _apply_spatial_filtering(self, speckle_pattern: np.ndarray) -> np.ndarray:
        """Apply spatial filtering to match speckle size."""
        # Create Gaussian filter kernel
        kernel_size = max(3, int(self.speckle_pixels))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = self.speckle_pixels / 3.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_2d = kernel * kernel.T
        
        # Apply filtering
        filtered_pattern = cv2.filter2D(speckle_pattern, -1, kernel_2d)
        
        return filtered_pattern
    
    def generate_temporal_sequence(
        self,
        num_frames: int = 10,
        temporal_correlation: float = 0.8,
        **kwargs
    ) -> np.ndarray:
        """
        Generate temporal sequence of speckle patterns.
        
        Args:
            num_frames: Number of temporal frames
            temporal_correlation: Correlation between consecutive frames
            **kwargs: Additional arguments for speckle generation
        
        Returns:
            Temporal sequence of speckle patterns
        """
        sequence = []
        
        # Generate first frame
        first_frame = self.generate_speckle_pattern(**kwargs)
        sequence.append(first_frame)
        
        # Generate subsequent frames with temporal correlation
        for i in range(1, num_frames):
            # Create correlated noise
            correlated_noise = temporal_correlation * sequence[i-1] + \
                             np.sqrt(1 - temporal_correlation**2) * np.random.random(self.image_size)
            
            # Generate new speckle pattern
            new_frame = self.generate_speckle_pattern(**kwargs)
            
            # Blend with correlated noise
            blended_frame = 0.7 * new_frame + 0.3 * correlated_noise
            sequence.append(blended_frame)
        
        return np.array(sequence)
    
    def calculate_contrast_map(
        self,
        temporal_sequence: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Calculate speckle contrast map from temporal sequence.
        
        Args:
            temporal_sequence: Temporal sequence of speckle patterns
            window_size: Window size for contrast calculation
        
        Returns:
            Contrast map
        """
        num_frames, height, width = temporal_sequence.shape
        contrast_map = np.zeros((height, width))
        
        # Calculate contrast for each pixel using temporal statistics
        for i in range(height):
            for j in range(width):
                pixel_sequence = temporal_sequence[:, i, j]
                
                # Calculate temporal mean and standard deviation
                mean_intensity = np.mean(pixel_sequence)
                std_intensity = np.std(pixel_sequence)
                
                # Calculate contrast
                if mean_intensity > 0:
                    contrast = std_intensity / mean_intensity
                else:
                    contrast = 0.0
                
                contrast_map[i, j] = contrast
        
        # Apply spatial smoothing
        contrast_map = cv2.GaussianBlur(contrast_map, (window_size, window_size), 1.0)
        
        return contrast_map
    
    def create_defect_region(
        self,
        defect_type: str,
        center: Tuple[int, int],
        radius: int,
        intensity_factor: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create a defect region specification.
        
        Args:
            defect_type: Type of defect
            center: Center coordinates (x, y)
            radius: Defect radius in pixels
            intensity_factor: Intensity modification factor
        
        Returns:
            Defect region dictionary
        """
        return {
            'type': defect_type,
            'center': center,
            'radius': radius,
            'intensity_factor': intensity_factor
        }
    
    def visualize_speckle_analysis(
        self,
        speckle_pattern: np.ndarray,
        contrast_map: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize speckle pattern and contrast analysis.
        
        Args:
            speckle_pattern: Speckle pattern image
            contrast_map: Contrast map (optional)
            save_path: Path to save visualization
        """
        if contrast_map is None:
            # Single image visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original speckle pattern
            axes[0].imshow(speckle_pattern, cmap='gray')
            axes[0].set_title('Laser Speckle Pattern')
            axes[0].axis('off')
            
            # Histogram
            axes[1].hist(speckle_pattern.flatten(), bins=50, density=True, alpha=0.7)
            axes[1].set_title('Intensity Distribution')
            axes[1].set_xlabel('Intensity')
            axes[1].set_ylabel('Density')
            
        else:
            # Dual image visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Original speckle pattern
            axes[0].imshow(speckle_pattern, cmap='gray')
            axes[0].set_title('Laser Speckle Pattern')
            axes[0].axis('off')
            
            # Contrast map
            im = axes[1].imshow(contrast_map, cmap='viridis')
            axes[1].set_title('Speckle Contrast Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Contrast histogram
            axes[2].hist(contrast_map.flatten(), bins=50, density=True, alpha=0.7)
            axes[2].set_title('Contrast Distribution')
            axes[2].set_xlabel('Contrast')
            axes[2].set_ylabel('Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Speckle analysis visualization saved to {save_path}")
        
        plt.show()

# Optimized functions using Numba
@jit(nopython=True, parallel=True)
def _fast_speckle_statistics(intensity_pattern: np.ndarray, contrast: float) -> np.ndarray:
    """Fast speckle statistics calculation using Numba."""
    height, width = intensity_pattern.shape
    result = np.zeros_like(intensity_pattern)
    
    shape_param = 1.0 / (contrast * contrast)
    
    for i in prange(height):
        for j in prange(width):
            scale_param = intensity_pattern[i, j] / shape_param
            result[i, j] = np.random.gamma(shape_param, scale_param)
    
    return result

def create_lsci_dataset(
    num_samples: int,
    image_size: Tuple[int, int] = (512, 512),
    defect_probability: float = 0.3,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of LSCI images with corresponding defect masks.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of generated images
        defect_probability: Probability of including defects
        **kwargs: Additional arguments for LSCI simulator
    
    Returns:
        Tuple of (images, masks)
    """
    simulator = LSCISimulator(image_size=image_size, **kwargs)
    
    images = []
    masks = []
    
    for i in range(num_samples):
        # Generate defect regions
        defect_regions = []
        mask = np.zeros(image_size, dtype=np.uint8)
        
        if np.random.random() < defect_probability:
            num_defects = np.random.randint(1, 4)
            
            for _ in range(num_defects):
                # Random defect properties
                defect_type = np.random.choice([
                    'surface_scratch', 'internal_crack', 
                    'material_inclusion', 'thermal_damage'
                ])
                
                center = (
                    np.random.randint(50, image_size[1] - 50),
                    np.random.randint(50, image_size[0] - 50)
                )
                radius = np.random.randint(10, 30)
                
                defect_regions.append(simulator.create_defect_region(
                    defect_type, center, radius
                ))
                
                # Create mask
                y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
                defect_mask = (x_coords - center[0])**2 + (y_coords - center[1])**2 <= radius**2
                mask[defect_mask] = 1
        
        # Generate speckle pattern
        speckle_pattern = simulator.generate_speckle_pattern(
            defect_regions=defect_regions
        )
        
        images.append(speckle_pattern)
        masks.append(mask)
    
    return np.array(images), np.array(masks)