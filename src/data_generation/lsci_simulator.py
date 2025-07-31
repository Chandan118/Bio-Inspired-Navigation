"""
Laser Speckle Contrast Imaging (LSCI) Simulator.
Generates realistic speckle patterns and contrast images for defect detection.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional, Dict, Any, List
import random
from numba import jit
import warnings


class LSCISimulator:
    """
    Advanced Laser Speckle Contrast Imaging simulator.
    Generates realistic speckle patterns with configurable parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSCI simulator with configuration.
        
        Args:
            config: Configuration dictionary containing LSCI parameters
        """
        self.wavelength = config.get('lsci_wavelength', 785.0)  # nm
        self.coherence_length = config.get('lsci_coherence_length', 50.0)  # μm
        self.speckle_size = config.get('lsci_speckle_size', 2.0)  # μm
        self.contrast_range = config.get('lsci_contrast_range', (0.1, 0.8))
        self.pixel_size = config.get('pixel_size', 1.0)  # μm
        self.exposure_time = config.get('exposure_time', 0.01)  # seconds
        
        # Derived parameters
        self.wavenumber = 2 * np.pi / (self.wavelength * 1e-9)  # m^-1
        self.coherence_time = self.coherence_length / (3e8)  # seconds
        
    def generate_speckle_pattern(
        self, 
        shape: Tuple[int, int],
        surface_roughness: float = 0.1,
        correlation_length: float = 10.0
    ) -> np.ndarray:
        """
        Generate realistic speckle pattern.
        
        Args:
            shape: Image shape (height, width)
            surface_roughness: Surface roughness in μm
            correlation_length: Surface correlation length in μm
            
        Returns:
            Speckle pattern image
        """
        height, width = shape
        
        # Generate surface height profile
        surface_profile = self._generate_surface_profile(
            shape, surface_roughness, correlation_length
        )
        
        # Calculate phase variations
        phase_variations = 2 * self.wavenumber * surface_profile * 1e-6  # Convert to m
        
        # Generate complex field
        complex_field = np.exp(1j * phase_variations)
        
        # Apply speckle size filter
        speckle_filter = self._create_speckle_filter(shape)
        filtered_field = ifft2(fft2(complex_field) * speckle_filter)
        
        # Calculate intensity
        intensity = np.abs(filtered_field) ** 2
        
        # Normalize
        intensity = intensity / np.max(intensity)
        
        return intensity
    
    def calculate_speckle_contrast(
        self, 
        speckle_pattern: np.ndarray,
        window_size: int = 7
    ) -> np.ndarray:
        """
        Calculate speckle contrast image.
        
        Args:
            speckle_pattern: Input speckle pattern
            window_size: Size of contrast calculation window
            
        Returns:
            Speckle contrast image
        """
        # Pad image for convolution
        pad_size = window_size // 2
        padded = np.pad(speckle_pattern, pad_size, mode='reflect')
        
        # Calculate local mean
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = convolve2d(padded, kernel, mode='valid')
        
        # Calculate local variance
        local_var = convolve2d(padded ** 2, kernel, mode='valid') - local_mean ** 2
        
        # Calculate contrast
        contrast = np.sqrt(local_var) / (local_mean + 1e-8)
        
        return contrast
    
    def generate_defect_speckle(
        self,
        base_speckle: np.ndarray,
        defect_mask: np.ndarray,
        defect_type: str = "surface_scratch"
    ) -> np.ndarray:
        """
        Generate speckle pattern with defects.
        
        Args:
            base_speckle: Base speckle pattern
            defect_mask: Binary defect mask
            defect_type: Type of defect
            
        Returns:
            Speckle pattern with defects
        """
        defect_speckle = base_speckle.copy()
        
        if defect_type == "surface_scratch":
            defect_speckle = self._apply_surface_scratch(defect_speckle, defect_mask)
        elif defect_type == "internal_crack":
            defect_speckle = self._apply_internal_crack(defect_speckle, defect_mask)
        elif defect_type == "material_inclusion":
            defect_speckle = self._apply_material_inclusion(defect_speckle, defect_mask)
        elif defect_type == "thermal_damage":
            defect_speckle = self._apply_thermal_damage(defect_speckle, defect_mask)
        elif defect_type == "corrosion":
            defect_speckle = self._apply_corrosion(defect_speckle, defect_mask)
        elif defect_type == "wear":
            defect_speckle = self._apply_wear(defect_speckle, defect_mask)
        
        return defect_speckle
    
    def generate_temporal_speckle(
        self,
        base_speckle: np.ndarray,
        num_frames: int = 10,
        motion_velocity: float = 1.0  # μm/s
    ) -> np.ndarray:
        """
        Generate temporal speckle sequence.
        
        Args:
            base_speckle: Base speckle pattern
            num_frames: Number of temporal frames
            motion_velocity: Motion velocity in μm/s
            
        Returns:
            Temporal speckle sequence
        """
        height, width = base_speckle.shape
        temporal_speckle = np.zeros((num_frames, height, width))
        
        for frame in range(num_frames):
            # Calculate displacement
            displacement = motion_velocity * frame * self.exposure_time
            
            # Apply motion
            motion_matrix = np.array([
                [1, 0, displacement],
                [0, 1, 0]
            ], dtype=np.float32)
            
            # Warp image
            warped = cv2.warpAffine(
                base_speckle, 
                motion_matrix, 
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            temporal_speckle[frame] = warped
        
        return temporal_speckle
    
    def calculate_temporal_contrast(
        self,
        temporal_speckle: np.ndarray,
        spatial_window: int = 7,
        temporal_window: int = 5
    ) -> np.ndarray:
        """
        Calculate temporal speckle contrast.
        
        Args:
            temporal_speckle: Temporal speckle sequence
            spatial_window: Spatial window size
            temporal_window: Temporal window size
            
        Returns:
            Temporal contrast image
        """
        num_frames, height, width = temporal_speckle.shape
        
        # Pad temporal sequence
        pad_frames = temporal_window // 2
        padded = np.pad(temporal_speckle, ((pad_frames, pad_frames), (0, 0), (0, 0)), mode='edge')
        
        # Calculate temporal statistics
        temporal_mean = np.mean(padded, axis=0)
        temporal_var = np.var(padded, axis=0)
        
        # Apply spatial smoothing
        kernel = np.ones((spatial_window, spatial_window)) / (spatial_window ** 2)
        smoothed_mean = convolve2d(temporal_mean, kernel, mode='same')
        smoothed_var = convolve2d(temporal_var, kernel, mode='same')
        
        # Calculate contrast
        contrast = np.sqrt(smoothed_var) / (smoothed_mean + 1e-8)
        
        return contrast
    
    def generate_multi_wavelength_speckle(
        self,
        shape: Tuple[int, int],
        wavelengths: List[float] = None
    ) -> np.ndarray:
        """
        Generate multi-wavelength speckle patterns.
        
        Args:
            shape: Image shape
            wavelengths: List of wavelengths in nm
            
        Returns:
            Multi-wavelength speckle pattern
        """
        if wavelengths is None:
            wavelengths = [785, 830, 850]  # Common LSCI wavelengths
        
        multi_speckle = np.zeros((len(wavelengths), *shape))
        
        for i, wavelength in enumerate(wavelengths):
            # Temporarily change wavelength
            original_wavelength = self.wavelength
            self.wavelength = wavelength
            self.wavenumber = 2 * np.pi / (wavelength * 1e-9)
            
            # Generate speckle for this wavelength
            multi_speckle[i] = self.generate_speckle_pattern(shape)
            
            # Restore original wavelength
            self.wavelength = original_wavelength
            self.wavenumber = 2 * np.pi / (original_wavelength * 1e-9)
        
        return multi_speckle
    
    def _generate_surface_profile(
        self,
        shape: Tuple[int, int],
        roughness: float,
        correlation_length: float
    ) -> np.ndarray:
        """
        Generate realistic surface height profile.
        
        Args:
            shape: Surface shape
            roughness: Surface roughness
            correlation_length: Surface correlation length
            
        Returns:
            Surface height profile
        """
        height, width = shape
        
        # Generate random surface
        surface = np.random.normal(0, roughness, shape)
        
        # Apply spatial correlation
        correlation_pixels = int(correlation_length / self.pixel_size)
        sigma = correlation_pixels / 3  # Gaussian filter parameter
        
        correlated_surface = ndimage.gaussian_filter(surface, sigma=sigma)
        
        return correlated_surface
    
    def _create_speckle_filter(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create speckle size filter in frequency domain.
        
        Args:
            shape: Image shape
            
        Returns:
            Speckle filter
        """
        height, width = shape
        
        # Create frequency grid
        fy = np.fft.fftfreq(height)
        fx = np.fft.fftfreq(width)
        fxx, fyy = np.meshgrid(fx, fy)
        
        # Calculate frequency magnitude
        freq_mag = np.sqrt(fxx**2 + fyy**2)
        
        # Speckle size in frequency domain
        speckle_freq = self.pixel_size / (self.speckle_size * 1e-6)
        
        # Create low-pass filter
        filter_response = np.exp(-(freq_mag / speckle_freq)**2)
        
        return fftshift(filter_response)
    
    def _apply_surface_scratch(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply surface scratch defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with surface scratch
        """
        # Surface scratch creates phase discontinuity
        scratch_depth = 0.5  # μm
        phase_shift = 2 * self.wavenumber * scratch_depth * 1e-6
        
        # Apply phase shift to defect region
        defect_speckle = speckle.copy()
        defect_speckle[defect_mask > 0] *= np.exp(1j * phase_shift)
        
        return np.abs(defect_speckle)
    
    def _apply_internal_crack(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply internal crack defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with internal crack
        """
        # Internal crack affects scattering
        defect_speckle = speckle.copy()
        
        # Reduce intensity in crack region
        intensity_reduction = 0.3
        defect_speckle[defect_mask > 0] *= (1 - intensity_reduction)
        
        # Add scattering artifacts
        scatter_noise = np.random.normal(0, 0.1, speckle.shape)
        defect_speckle[defect_mask > 0] += scatter_noise[defect_mask > 0]
        
        return np.clip(defect_speckle, 0, 1)
    
    def _apply_material_inclusion(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply material inclusion defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with material inclusion
        """
        # Material inclusion changes refractive index
        defect_speckle = speckle.copy()
        
        # Simulate different refractive index
        refractive_index_change = 0.1
        phase_change = 2 * self.wavenumber * refractive_index_change * 1e-6
        
        # Apply phase change
        defect_speckle[defect_mask > 0] *= np.exp(1j * phase_change)
        
        return np.abs(defect_speckle)
    
    def _apply_thermal_damage(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply thermal damage defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with thermal damage
        """
        # Thermal damage creates surface deformation
        defect_speckle = speckle.copy()
        
        # Create thermal deformation pattern
        deformation = ndimage.gaussian_filter(defect_mask.astype(float), sigma=3)
        
        # Apply deformation effect
        defect_speckle[defect_mask > 0] *= (1 + 0.2 * deformation[defect_mask > 0])
        
        # Add thermal artifacts
        thermal_noise = np.random.normal(0, 0.15, speckle.shape)
        defect_speckle[defect_mask > 0] += thermal_noise[defect_mask > 0]
        
        return np.clip(defect_speckle, 0, 1)
    
    def _apply_corrosion(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply corrosion defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with corrosion
        """
        # Corrosion creates rough surface
        defect_speckle = speckle.copy()
        
        # Create corrosion pattern
        corrosion_pattern = np.random.normal(0, 0.2, speckle.shape)
        corrosion_pattern = ndimage.gaussian_filter(corrosion_pattern, sigma=2)
        
        # Apply corrosion effect
        defect_speckle[defect_mask > 0] *= (1 + corrosion_pattern[defect_mask > 0])
        
        # Add oxidation artifacts
        oxidation_noise = np.random.exponential(0.1, speckle.shape)
        defect_speckle[defect_mask > 0] += oxidation_noise[defect_mask > 0]
        
        return np.clip(defect_speckle, 0, 1)
    
    def _apply_wear(
        self,
        speckle: np.ndarray,
        defect_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply wear defect.
        
        Args:
            speckle: Input speckle pattern
            defect_mask: Defect mask
            
        Returns:
            Speckle with wear
        """
        # Wear creates gradual surface changes
        defect_speckle = speckle.copy()
        
        # Create wear pattern
        wear_gradient = ndimage.gaussian_filter(defect_mask.astype(float), sigma=5)
        
        # Apply wear effect
        wear_intensity = 0.4
        defect_speckle[defect_mask > 0] *= (1 - wear_intensity * wear_gradient[defect_mask > 0])
        
        # Add wear artifacts
        wear_noise = np.random.normal(0, 0.1, speckle.shape)
        defect_speckle[defect_mask > 0] += wear_noise[defect_mask > 0]
        
        return np.clip(defect_speckle, 0, 1)


@jit(nopython=True)
def fast_speckle_contrast(speckle: np.ndarray, window_size: int) -> np.ndarray:
    """
    Fast speckle contrast calculation using Numba.
    
    Args:
        speckle: Input speckle pattern
        window_size: Window size for contrast calculation
        
    Returns:
        Speckle contrast image
    """
    height, width = speckle.shape
    contrast = np.zeros((height, width))
    
    pad = window_size // 2
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract window
            window = speckle[i-pad:i+pad+1, j-pad:j+pad+1]
            
            # Calculate mean and variance
            mean_val = np.mean(window)
            var_val = np.var(window)
            
            # Calculate contrast
            if mean_val > 1e-8:
                contrast[i, j] = np.sqrt(var_val) / mean_val
            else:
                contrast[i, j] = 0
    
    return contrast


def create_lsci_config(
    wavelength: float = 785.0,
    speckle_size: float = 2.0,
    contrast_range: Tuple[float, float] = (0.1, 0.8)
) -> Dict[str, Any]:
    """
    Create LSCI configuration dictionary.
    
    Args:
        wavelength: Laser wavelength in nm
        speckle_size: Speckle size in μm
        contrast_range: Contrast range (min, max)
        
    Returns:
        LSCI configuration dictionary
    """
    return {
        'lsci_wavelength': wavelength,
        'lsci_coherence_length': 50.0,
        'lsci_speckle_size': speckle_size,
        'lsci_contrast_range': contrast_range,
        'pixel_size': 1.0,
        'exposure_time': 0.01
    }