"""
Noise models for realistic industrial environment simulation.
Implements various noise types commonly encountered in optical imaging systems.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Tuple, Optional, Dict, Any
import random
from numba import jit
import warnings


class NoiseModel:
    """
    Base class for noise models used in optical imaging simulation.
    Provides realistic noise patterns for industrial environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize noise model with configuration.
        
        Args:
            config: Configuration dictionary containing noise parameters
        """
        self.config = config
        self.thermal_noise_std = config.get('thermal_noise_std', 0.05)
        self.motion_blur_probability = config.get('motion_blur_probability', 0.1)
        self.motion_blur_kernel_size = config.get('motion_blur_kernel_size', 15)
        self.gaussian_noise_std = config.get('gaussian_noise_std', 0.02)
        self.salt_pepper_probability = config.get('salt_pepper_probability', 0.01)
        
    def apply_thermal_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply thermal noise to simulate sensor heating effects.
        
        Args:
            image: Input image
            
        Returns:
            Image with thermal noise
        """
        # Generate thermal noise with spatial correlation
        noise = self._generate_correlated_noise(
            image.shape, 
            self.thermal_noise_std,
            correlation_length=5
        )
        
        # Add temperature-dependent intensity variation
        temp_gradient = self._create_temperature_gradient(image.shape)
        thermal_effect = noise * (1 + 0.3 * temp_gradient)
        
        noisy_image = image + thermal_effect
        return np.clip(noisy_image, 0, 1)
    
    def apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply motion blur to simulate vibration and movement.
        
        Args:
            image: Input image
            
        Returns:
            Motion blurred image
        """
        if random.random() > self.motion_blur_probability:
            return image
        
        # Random motion parameters
        angle = random.uniform(0, 360)
        distance = random.uniform(5, self.motion_blur_kernel_size)
        
        # Create motion blur kernel
        kernel = self._create_motion_kernel(angle, distance)
        
        # Apply convolution
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred[:, :, channel] = convolve2d(
                    image[:, :, channel], kernel, mode='same', boundary='symm'
                )
        else:
            blurred = convolve2d(image, kernel, mode='same', boundary='symm')
        
        return blurred
    
    def apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to simulate electronic noise.
        
        Args:
            image: Input image
            
        Returns:
            Image with Gaussian noise
        """
        noise = np.random.normal(0, self.gaussian_noise_std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_salt_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply salt and pepper noise to simulate sensor defects.
        
        Args:
            image: Input image
            
        Returns:
            Image with salt and pepper noise
        """
        noisy_image = image.copy()
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(image.shape) < self.salt_pepper_probability / 2
        noisy_image[salt_mask] = 1
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape) < self.salt_pepper_probability / 2
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def apply_quantization_noise(self, image: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Apply quantization noise to simulate limited bit depth.
        
        Args:
            image: Input image
            bits: Number of bits for quantization
            
        Returns:
            Quantized image
        """
        levels = 2 ** bits
        quantized = np.round(image * (levels - 1)) / (levels - 1)
        return quantized
    
    def apply_shot_noise(self, image: np.ndarray, photon_count: float = 1000) -> np.ndarray:
        """
        Apply shot noise to simulate photon counting statistics.
        
        Args:
            image: Input image
            photon_count: Average photon count
            
        Returns:
            Image with shot noise
        """
        # Convert to photon counts
        photon_image = image * photon_count
        
        # Apply Poisson noise
        noisy_photons = np.random.poisson(photon_image)
        
        # Convert back to intensity
        noisy_image = noisy_photons / photon_count
        return np.clip(noisy_image, 0, 1)
    
    def apply_dark_current_noise(self, image: np.ndarray, dark_current: float = 0.1) -> np.ndarray:
        """
        Apply dark current noise to simulate sensor dark current.
        
        Args:
            image: Input image
            dark_current: Dark current level
            
        Returns:
            Image with dark current noise
        """
        dark_noise = np.random.exponential(dark_current, image.shape)
        noisy_image = image + dark_noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_readout_noise(self, image: np.ndarray, readout_std: float = 0.02) -> np.ndarray:
        """
        Apply readout noise to simulate amplifier noise.
        
        Args:
            image: Input image
            readout_std: Readout noise standard deviation
            
        Returns:
            Image with readout noise
        """
        readout_noise = np.random.normal(0, readout_std, image.shape)
        noisy_image = image + readout_noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_banding_noise(self, image: np.ndarray, band_width: int = 10) -> np.ndarray:
        """
        Apply banding noise to simulate sensor readout artifacts.
        
        Args:
            image: Input image
            band_width: Width of noise bands
            
        Returns:
            Image with banding noise
        """
        height, width = image.shape[:2]
        banding = np.zeros_like(image)
        
        for i in range(0, height, band_width):
            band_intensity = np.random.normal(0, 0.05)
            banding[i:i+band_width, :] = band_intensity
        
        noisy_image = image + banding
        return np.clip(noisy_image, 0, 1)
    
    def apply_uneven_illumination(self, image: np.ndarray, intensity_variation: float = 0.3) -> np.ndarray:
        """
        Apply uneven illumination to simulate non-uniform lighting.
        
        Args:
            image: Input image
            intensity_variation: Maximum intensity variation
            
        Returns:
            Image with uneven illumination
        """
        height, width = image.shape[:2]
        
        # Create radial illumination pattern
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        
        # Radial distance from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_r = np.sqrt(center_x**2 + center_y**2)
        
        # Create illumination mask
        illumination = 1 - intensity_variation * (r / max_r)**2
        illumination = np.clip(illumination, 1 - intensity_variation, 1)
        
        # Apply to image
        if len(image.shape) == 3:
            illumination = np.stack([illumination] * image.shape[2], axis=2)
        
        illuminated_image = image * illumination
        return illuminated_image
    
    def apply_optical_distortion(self, image: np.ndarray, distortion_strength: float = 0.1) -> np.ndarray:
        """
        Apply optical distortion to simulate lens aberrations.
        
        Args:
            image: Input image
            distortion_strength: Strength of distortion
            
        Returns:
            Distorted image
        """
        height, width = image.shape[:2]
        center_y, center_x = height // 2, width // 2
        
        # Create coordinate grids
        y, x = np.mgrid[:height, :width]
        
        # Radial distortion
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_r = np.sqrt(center_x**2 + center_y**2)
        
        # Distortion factor
        distortion_factor = 1 + distortion_strength * (r / max_r)**2
        
        # Apply distortion
        x_distorted = center_x + (x - center_x) * distortion_factor
        y_distorted = center_y + (y - center_y) * distortion_factor
        
        # Interpolate
        distorted_image = ndimage.map_coordinates(
            image, 
            [y_distorted, x_distorted], 
            order=1, 
            mode='reflect'
        )
        
        return distorted_image
    
    def apply_comprehensive_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a comprehensive combination of all noise types.
        
        Args:
            image: Input image
            
        Returns:
            Image with comprehensive noise
        """
        # Apply noise in sequence
        noisy_image = image.copy()
        
        # Optical effects first
        noisy_image = self.apply_uneven_illumination(noisy_image)
        noisy_image = self.apply_optical_distortion(noisy_image)
        
        # Motion effects
        noisy_image = self.apply_motion_blur(noisy_image)
        
        # Sensor noise
        noisy_image = self.apply_thermal_noise(noisy_image)
        noisy_image = self.apply_shot_noise(noisy_image)
        noisy_image = self.apply_dark_current_noise(noisy_image)
        noisy_image = self.apply_readout_noise(noisy_image)
        noisy_image = self.apply_gaussian_noise(noisy_image)
        
        # Artifacts
        noisy_image = self.apply_salt_pepper_noise(noisy_image)
        noisy_image = self.apply_banding_noise(noisy_image)
        
        # Quantization
        noisy_image = self.apply_quantization_noise(noisy_image)
        
        return noisy_image
    
    def _generate_correlated_noise(self, shape: Tuple[int, ...], std: float, correlation_length: int) -> np.ndarray:
        """
        Generate spatially correlated noise.
        
        Args:
            shape: Shape of the noise array
            std: Standard deviation
            correlation_length: Spatial correlation length
            
        Returns:
            Correlated noise array
        """
        # Generate white noise
        white_noise = np.random.normal(0, 1, shape)
        
        # Apply spatial correlation using Gaussian filter
        correlated_noise = ndimage.gaussian_filter(
            white_noise, 
            sigma=correlation_length
        )
        
        # Normalize to desired standard deviation
        correlated_noise = correlated_noise * std / np.std(correlated_noise)
        
        return correlated_noise
    
    def _create_temperature_gradient(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create temperature gradient for thermal noise simulation.
        
        Args:
            shape: Shape of the gradient array
            
        Returns:
            Temperature gradient array
        """
        height, width = shape[:2]
        
        # Create radial temperature gradient
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_r = np.sqrt(center_x**2 + center_y**2)
        
        # Temperature increases towards edges
        temperature = r / max_r
        
        if len(shape) == 3:
            temperature = np.stack([temperature] * shape[2], axis=2)
        
        return temperature
    
    def _create_motion_kernel(self, angle: float, distance: float) -> np.ndarray:
        """
        Create motion blur kernel.
        
        Args:
            angle: Motion angle in degrees
            distance: Motion distance in pixels
            
        Returns:
            Motion blur kernel
        """
        kernel_size = int(max(distance * 2, self.motion_blur_kernel_size))
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create line in kernel
        for i in range(int(distance)):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        
        return kernel


class AdvancedNoiseModel(NoiseModel):
    """
    Advanced noise model with additional industrial-specific noise patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced noise model."""
        super().__init__(config)
        self.vibration_frequency = config.get('vibration_frequency', 50)  # Hz
        self.ambient_temperature = config.get('ambient_temperature', 25)  # Celsius
        self.humidity = config.get('humidity', 0.5)  # Relative humidity
        
    def apply_vibration_noise(self, image: np.ndarray, exposure_time: float = 0.01) -> np.ndarray:
        """
        Apply vibration-induced noise based on exposure time.
        
        Args:
            image: Input image
            exposure_time: Exposure time in seconds
            
        Returns:
            Image with vibration noise
        """
        # Vibration amplitude depends on frequency and exposure time
        vibration_amplitude = 0.1 * np.sin(2 * np.pi * self.vibration_frequency * exposure_time)
        
        # Create vibration pattern
        height, width = image.shape[:2]
        y, x = np.mgrid[:height, :width]
        
        # Vibration pattern
        vibration_pattern = vibration_amplitude * np.sin(2 * np.pi * x / width)
        
        if len(image.shape) == 3:
            vibration_pattern = np.stack([vibration_pattern] * image.shape[2], axis=2)
        
        noisy_image = image + vibration_pattern
        return np.clip(noisy_image, 0, 1)
    
    def apply_humidity_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply humidity-related noise to simulate moisture effects.
        
        Args:
            image: Input image
            
        Returns:
            Image with humidity noise
        """
        # Humidity affects image quality
        humidity_factor = 1 + 0.2 * (self.humidity - 0.5)
        
        # Add moisture-related artifacts
        moisture_noise = np.random.normal(0, 0.05 * humidity_factor, image.shape)
        
        noisy_image = image + moisture_noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_temperature_dependent_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply temperature-dependent noise patterns.
        
        Args:
            image: Input image
            
        Returns:
            Image with temperature-dependent noise
        """
        # Temperature affects noise characteristics
        temp_factor = (self.ambient_temperature - 20) / 30  # Normalized temperature
        
        # Adjust noise parameters based on temperature
        thermal_std = self.thermal_noise_std * (1 + 0.5 * temp_factor)
        gaussian_std = self.gaussian_noise_std * (1 + 0.3 * temp_factor)
        
        # Apply temperature-adjusted noise
        noisy_image = self.apply_thermal_noise(image)
        noisy_image = self.apply_gaussian_noise(noisy_image)
        
        return noisy_image


@jit(nopython=True)
def fast_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """
    Fast Gaussian noise generation using Numba.
    
    Args:
        image: Input image
        std: Standard deviation
        
    Returns:
        Image with Gaussian noise
    """
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def create_noise_profile(noise_type: str, intensity: float = 1.0) -> Dict[str, Any]:
    """
    Create noise profile for different imaging conditions.
    
    Args:
        noise_type: Type of noise profile ('low', 'medium', 'high', 'industrial')
        intensity: Noise intensity multiplier
        
    Returns:
        Noise configuration dictionary
    """
    profiles = {
        'low': {
            'thermal_noise_std': 0.02 * intensity,
            'gaussian_noise_std': 0.01 * intensity,
            'salt_pepper_probability': 0.005 * intensity,
            'motion_blur_probability': 0.05 * intensity,
        },
        'medium': {
            'thermal_noise_std': 0.05 * intensity,
            'gaussian_noise_std': 0.02 * intensity,
            'salt_pepper_probability': 0.01 * intensity,
            'motion_blur_probability': 0.1 * intensity,
        },
        'high': {
            'thermal_noise_std': 0.1 * intensity,
            'gaussian_noise_std': 0.05 * intensity,
            'salt_pepper_probability': 0.02 * intensity,
            'motion_blur_probability': 0.2 * intensity,
        },
        'industrial': {
            'thermal_noise_std': 0.08 * intensity,
            'gaussian_noise_std': 0.03 * intensity,
            'salt_pepper_probability': 0.015 * intensity,
            'motion_blur_probability': 0.15 * intensity,
            'vibration_frequency': 60,
            'ambient_temperature': 35,
            'humidity': 0.7,
        }
    }
    
    return profiles.get(noise_type, profiles['medium'])