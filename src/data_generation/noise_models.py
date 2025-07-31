"""
Noise models for realistic industrial environment simulation.
Implements various noise types encountered in real-world optical diagnostics.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import poisson, norm, gamma
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class NoiseModel:
    """Base class for noise models."""
    
    def __init__(self, intensity: float = 1.0):
        self.intensity = intensity
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply noise to image. To be implemented by subclasses."""
        raise NotImplementedError

class ThermalNoise(NoiseModel):
    """Thermal noise simulation for CCD/CMOS sensors."""
    
    def __init__(self, std_dev: float = 0.05, temperature: float = 300.0):
        super().__init__(intensity=std_dev)
        self.temperature = temperature  # Kelvin
        self.std_dev = std_dev
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply thermal noise based on temperature and sensor characteristics."""
        # Temperature-dependent noise variance
        # Higher temperature = more thermal noise
        temp_factor = np.sqrt(self.temperature / 300.0)
        effective_std = self.std_dev * temp_factor
        
        # Generate thermal noise
        noise = np.random.normal(0, effective_std, image.shape)
        
        # Add to image
        noisy_image = image + noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image

class ShotNoise(NoiseModel):
    """Shot noise (Poisson noise) simulation."""
    
    def __init__(self, photon_count_factor: float = 1000.0):
        super().__init__(intensity=photon_count_factor)
        self.photon_count_factor = photon_count_factor
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply shot noise based on photon statistics."""
        # Convert image intensity to photon counts
        photon_counts = image * self.photon_count_factor
        
        # Add Poisson noise
        noisy_photon_counts = np.random.poisson(photon_counts)
        
        # Convert back to intensity
        noisy_image = noisy_photon_counts / self.photon_count_factor
        
        return noisy_image

class MotionBlur(NoiseModel):
    """Motion blur simulation for vibration and movement."""
    
    def __init__(self, kernel_size: int = 15, angle: float = 0.0, velocity: float = 1.0):
        super().__init__(intensity=velocity)
        self.kernel_size = kernel_size
        self.angle = angle
        self.velocity = velocity
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply motion blur with random direction and intensity."""
        # Random blur angle if not specified
        if self.angle == 0.0:
            blur_angle = np.random.uniform(0, 2 * np.pi)
        else:
            blur_angle = self.angle
        
        # Random kernel size based on velocity
        effective_kernel_size = int(self.kernel_size * self.velocity)
        effective_kernel_size = max(3, min(effective_kernel_size, 31))  # Reasonable bounds
        
        # Create motion blur kernel
        kernel = np.zeros((effective_kernel_size, effective_kernel_size))
        center = effective_kernel_size // 2
        
        # Create line kernel
        for i in range(effective_kernel_size):
            for j in range(effective_kernel_size):
                # Calculate distance from center along blur direction
                dx = i - center
                dy = j - center
                distance = dx * np.cos(blur_angle) + dy * np.sin(blur_angle)
                
                # Create Gaussian profile along blur direction
                if abs(distance) <= effective_kernel_size // 4:
                    kernel[i, j] = np.exp(-(distance ** 2) / (2 * (effective_kernel_size // 8) ** 2))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply blur
        blurred_image = cv2.filter2D(image, -1, kernel)
        
        return blurred_image

class VibrationNoise(NoiseModel):
    """Vibration-induced noise simulation."""
    
    def __init__(self, amplitude: float = 2.0, frequency: float = 50.0):
        super().__init__(intensity=amplitude)
        self.amplitude = amplitude
        self.frequency = frequency
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply vibration-induced geometric distortion."""
        height, width = image.shape[:2]
        
        # Create displacement field
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Add vibration displacement
        time_factor = np.random.uniform(0, 2 * np.pi)
        displacement_x = self.amplitude * np.sin(2 * np.pi * self.frequency * time_factor + x_coords / 100)
        displacement_y = self.amplitude * np.cos(2 * np.pi * self.frequency * time_factor + y_coords / 100)
        
        # Apply displacement
        new_x = x_coords + displacement_x
        new_y = y_coords + displacement_y
        
        # Ensure coordinates are within bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Interpolate image
        if len(image.shape) == 3:
            distorted_image = np.zeros_like(image)
            for channel in range(image.shape[2]):
                distorted_image[:, :, channel] = ndimage.map_coordinates(
                    image[:, :, channel], [new_y, new_x], order=1
                )
        else:
            distorted_image = ndimage.map_coordinates(image, [new_y, new_x], order=1)
        
        return distorted_image

class ElectronicNoise(NoiseModel):
    """Electronic noise simulation (readout noise, quantization noise)."""
    
    def __init__(self, readout_noise_std: float = 0.02, bit_depth: int = 12):
        super().__init__(intensity=readout_noise_std)
        self.readout_noise_std = readout_noise_std
        self.bit_depth = bit_depth
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply electronic noise including readout and quantization noise."""
        # Readout noise
        readout_noise = np.random.normal(0, self.readout_noise_std, image.shape)
        
        # Quantization noise
        max_value = 2 ** self.bit_depth - 1
        quantization_noise = np.random.uniform(-0.5 / max_value, 0.5 / max_value, image.shape)
        
        # Combine noises
        total_noise = readout_noise + quantization_noise
        
        # Apply to image
        noisy_image = image + total_noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image

class AtmosphericTurbulence(NoiseModel):
    """Atmospheric turbulence simulation for long-range imaging."""
    
    def __init__(self, turbulence_strength: float = 0.1, correlation_length: float = 50.0):
        super().__init__(intensity=turbulence_strength)
        self.turbulence_strength = turbulence_strength
        self.correlation_length = correlation_length
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply atmospheric turbulence effects."""
        height, width = image.shape[:2]
        
        # Create phase screen
        phase_screen = self._generate_phase_screen(height, width)
        
        # Apply phase distortion
        if len(image.shape) == 3:
            distorted_image = np.zeros_like(image)
            for channel in range(image.shape[2]):
                distorted_image[:, :, channel] = self._apply_phase_distortion(
                    image[:, :, channel], phase_screen
                )
        else:
            distorted_image = self._apply_phase_distortion(image, phase_screen)
        
        return distorted_image
    
    def _generate_phase_screen(self, height: int, width: int) -> np.ndarray:
        """Generate Kolmogorov phase screen for atmospheric turbulence."""
        # Create frequency grid
        fx = np.fft.fftfreq(width)
        fy = np.fft.fftfreq(height)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        
        # Kolmogorov power spectrum
        freq_magnitude = np.sqrt(fx_grid**2 + fy_grid**2)
        freq_magnitude[0, 0] = 1e-10  # Avoid division by zero
        
        # Power spectrum
        power_spectrum = (freq_magnitude ** (-11/6)) * self.turbulence_strength
        
        # Generate random phase
        random_phase = np.random.uniform(0, 2 * np.pi, (height, width))
        
        # Create complex field
        complex_field = np.sqrt(power_spectrum) * np.exp(1j * random_phase)
        
        # Inverse FFT to get phase screen
        phase_screen = np.real(np.fft.ifft2(complex_field))
        
        return phase_screen
    
    def _apply_phase_distortion(self, image: np.ndarray, phase_screen: np.ndarray) -> np.ndarray:
        """Apply phase distortion to image."""
        # Normalize phase screen
        phase_screen = (phase_screen - np.min(phase_screen)) / (np.max(phase_screen) - np.min(phase_screen))
        
        # Apply distortion
        distorted_image = image * (1 - 0.3 * phase_screen)
        
        return distorted_image

class MultiNoiseModel:
    """Combined noise model that applies multiple noise types."""
    
    def __init__(self, noise_models: Dict[str, NoiseModel], probabilities: Optional[Dict[str, float]] = None):
        self.noise_models = noise_models
        self.probabilities = probabilities or {name: 1.0 for name in noise_models.keys()}
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply multiple noise models based on probabilities."""
        noisy_image = image.copy()
        
        for name, noise_model in self.noise_models.items():
            if np.random.random() < self.probabilities[name]:
                noisy_image = noise_model.apply(noisy_image)
                logger.debug(f"Applied {name} noise")
        
        return noisy_image

def create_industrial_noise_model() -> MultiNoiseModel:
    """Create a realistic industrial noise model."""
    noise_models = {
        'thermal': ThermalNoise(std_dev=0.05, temperature=320.0),
        'shot': ShotNoise(photon_count_factor=800.0),
        'motion_blur': MotionBlur(kernel_size=10, velocity=0.8),
        'vibration': VibrationNoise(amplitude=1.5, frequency=60.0),
        'electronic': ElectronicNoise(readout_noise_std=0.03, bit_depth=12),
        'atmospheric': AtmosphericTurbulence(turbulence_strength=0.05, correlation_length=30.0)
    }
    
    # Realistic probabilities for industrial environment
    probabilities = {
        'thermal': 1.0,      # Always present
        'shot': 1.0,         # Always present
        'motion_blur': 0.3,  # 30% chance
        'vibration': 0.4,    # 40% chance
        'electronic': 1.0,   # Always present
        'atmospheric': 0.1   # 10% chance (indoor/outdoor)
    }
    
    return MultiNoiseModel(noise_models, probabilities)

def create_laboratory_noise_model() -> MultiNoiseModel:
    """Create a controlled laboratory noise model."""
    noise_models = {
        'thermal': ThermalNoise(std_dev=0.02, temperature=295.0),
        'shot': ShotNoise(photon_count_factor=1200.0),
        'electronic': ElectronicNoise(readout_noise_std=0.01, bit_depth=14)
    }
    
    # Lower probabilities for controlled environment
    probabilities = {
        'thermal': 1.0,
        'shot': 1.0,
        'electronic': 1.0
    }
    
    return MultiNoiseModel(noise_models, probabilities)

def create_outdoor_noise_model() -> MultiNoiseModel:
    """Create an outdoor environment noise model."""
    noise_models = {
        'thermal': ThermalNoise(std_dev=0.08, temperature=310.0),
        'shot': ShotNoise(photon_count_factor=600.0),
        'motion_blur': MotionBlur(kernel_size=20, velocity=1.2),
        'vibration': VibrationNoise(amplitude=3.0, frequency=40.0),
        'electronic': ElectronicNoise(readout_noise_std=0.05, bit_depth=10),
        'atmospheric': AtmosphericTurbulence(turbulence_strength=0.15, correlation_length=20.0)
    }
    
    # Higher probabilities for outdoor environment
    probabilities = {
        'thermal': 1.0,
        'shot': 1.0,
        'motion_blur': 0.6,
        'vibration': 0.7,
        'electronic': 1.0,
        'atmospheric': 0.8
    }
    
    return MultiNoiseModel(noise_models, probabilities)