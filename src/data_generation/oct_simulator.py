"""
Optical Coherence Tomography (OCT) Simulator.
Generates realistic OCT images with depth-resolved information and defect simulation.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftshift, ifftshift
from typing import Tuple, Optional, Dict, Any, List
import random
from numba import jit
import warnings


class OCTSimulator:
    """
    Advanced Optical Coherence Tomography simulator.
    Generates realistic OCT images with depth-resolved information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OCT simulator with configuration.
        
        Args:
            config: Configuration dictionary containing OCT parameters
        """
        self.center_wavelength = config.get('oct_center_wavelength', 1300.0)  # nm
        self.bandwidth = config.get('oct_bandwidth', 100.0)  # nm
        self.axial_resolution = config.get('oct_axial_resolution', 5.0)  # μm
        self.lateral_resolution = config.get('oct_lateral_resolution', 10.0)  # μm
        self.depth_range = config.get('oct_depth_range', 2000.0)  # μm
        self.pixel_size = config.get('pixel_size', 1.0)  # μm
        
        # Derived parameters
        self.coherence_length = (self.center_wavelength ** 2) / (2 * self.bandwidth)  # nm
        self.depth_pixels = int(self.depth_range / self.axial_resolution)
        self.sampling_rate = 2 * self.bandwidth / (3e8 * 1e-9)  # Hz
        
    def generate_oct_image(
        self,
        shape: Tuple[int, int],
        material_properties: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Generate realistic OCT image.
        
        Args:
            shape: Image shape (depth, width)
            material_properties: Material optical properties
            
        Returns:
            OCT image
        """
        if material_properties is None:
            material_properties = {
                'refractive_index': 1.5,
                'scattering_coefficient': 100.0,  # cm^-1
                'absorption_coefficient': 1.0,    # cm^-1
                'anisotropy_factor': 0.9
            }
        
        depth, width = shape
        
        # Generate depth-resolved reflectivity profile
        reflectivity_profile = self._generate_reflectivity_profile(
            shape, material_properties
        )
        
        # Apply depth-dependent attenuation
        attenuated_profile = self._apply_depth_attenuation(
            reflectivity_profile, material_properties
        )
        
        # Generate OCT A-scan for each lateral position
        oct_image = np.zeros(shape)
        
        for x in range(width):
            # Generate A-scan
            a_scan = self._generate_a_scan(
                attenuated_profile[:, x], 
                material_properties
            )
            oct_image[:, x] = a_scan
        
        # Apply lateral smoothing
        oct_image = ndimage.gaussian_filter(oct_image, sigma=(0, 1))
        
        # Normalize
        oct_image = oct_image / np.max(oct_image)
        
        return oct_image
    
    def generate_defect_oct(
        self,
        base_oct: np.ndarray,
        defect_mask: np.ndarray,
        defect_type: str = "internal_crack",
        defect_depth: float = 500.0  # μm
    ) -> np.ndarray:
        """
        Generate OCT image with defects.
        
        Args:
            base_oct: Base OCT image
            defect_mask: Binary defect mask
            defect_type: Type of defect
            defect_depth: Depth of defect in μm
            
        Returns:
            OCT image with defects
        """
        defect_oct = base_oct.copy()
        depth, width = base_oct.shape
        
        # Convert depth to pixel index
        defect_pixel = int(defect_depth / self.axial_resolution)
        defect_pixel = min(defect_pixel, depth - 1)
        
        if defect_type == "internal_crack":
            defect_oct = self._apply_internal_crack_oct(defect_oct, defect_mask, defect_pixel)
        elif defect_type == "material_inclusion":
            defect_oct = self._apply_material_inclusion_oct(defect_oct, defect_mask, defect_pixel)
        elif defect_type == "delamination":
            defect_oct = self._apply_delamination_oct(defect_oct, defect_mask, defect_pixel)
        elif defect_type == "porosity":
            defect_oct = self._apply_porosity_oct(defect_oct, defect_mask, defect_pixel)
        elif defect_type == "thermal_damage":
            defect_oct = self._apply_thermal_damage_oct(defect_oct, defect_mask, defect_pixel)
        
        return defect_oct
    
    def generate_3d_oct_volume(
        self,
        volume_shape: Tuple[int, int, int],
        material_properties: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Generate 3D OCT volume.
        
        Args:
            volume_shape: Volume shape (depth, height, width)
            material_properties: Material optical properties
            
        Returns:
            3D OCT volume
        """
        depth, height, width = volume_shape
        volume = np.zeros(volume_shape)
        
        for y in range(height):
            # Generate 2D OCT slice
            slice_2d = self.generate_oct_image((depth, width), material_properties)
            volume[:, y, :] = slice_2d
        
        return volume
    
    def calculate_oct_metrics(
        self,
        oct_image: np.ndarray,
        region_of_interest: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, float]:
        """
        Calculate OCT image quality metrics.
        
        Args:
            oct_image: OCT image
            region_of_interest: ROI coordinates (top, bottom, left, right)
            
        Returns:
            Dictionary of OCT metrics
        """
        if region_of_interest is not None:
            top, bottom, left, right = region_of_interest
            oct_roi = oct_image[top:bottom, left:right]
        else:
            oct_roi = oct_image
        
        # Calculate metrics
        metrics = {
            'mean_intensity': np.mean(oct_roi),
            'std_intensity': np.std(oct_roi),
            'contrast': np.std(oct_roi) / (np.mean(oct_roi) + 1e-8),
            'snr': np.mean(oct_roi) / (np.std(oct_roi) + 1e-8),
            'dynamic_range': np.max(oct_roi) - np.min(oct_roi),
            'entropy': self._calculate_entropy(oct_roi)
        }
        
        return metrics
    
    def apply_oct_noise(
        self,
        oct_image: np.ndarray,
        noise_type: str = "shot_noise",
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Apply OCT-specific noise to image.
        
        Args:
            oct_image: Input OCT image
            noise_type: Type of noise
            noise_level: Noise intensity
            
        Returns:
            Noisy OCT image
        """
        if noise_type == "shot_noise":
            return self._apply_shot_noise(oct_image, noise_level)
        elif noise_type == "speckle_noise":
            return self._apply_speckle_noise(oct_image, noise_level)
        elif noise_type == "electronic_noise":
            return self._apply_electronic_noise(oct_image, noise_level)
        elif noise_type == "motion_artifact":
            return self._apply_motion_artifact(oct_image, noise_level)
        else:
            return oct_image
    
    def _generate_reflectivity_profile(
        self,
        shape: Tuple[int, int],
        material_properties: Dict[str, float]
    ) -> np.ndarray:
        """
        Generate depth-resolved reflectivity profile.
        
        Args:
            shape: Image shape
            material_properties: Material properties
            
        Returns:
            Reflectivity profile
        """
        depth, width = shape
        profile = np.zeros(shape)
        
        # Generate surface layer
        surface_reflectivity = 0.1
        surface_thickness = int(10 / self.axial_resolution)  # 10 μm surface layer
        
        profile[:surface_thickness, :] = surface_reflectivity
        
        # Generate bulk material with random variations
        bulk_reflectivity = 0.05
        for x in range(width):
            # Create depth-dependent reflectivity
            depth_variation = np.random.normal(0, 0.02, depth - surface_thickness)
            profile[surface_thickness:, x] = bulk_reflectivity + depth_variation
        
        # Add random interfaces
        num_interfaces = random.randint(2, 5)
        for _ in range(num_interfaces):
            interface_depth = random.randint(surface_thickness, depth - 10)
            interface_width = random.randint(width // 4, width // 2)
            interface_start = random.randint(0, width - interface_width)
            
            interface_reflectivity = random.uniform(0.1, 0.3)
            profile[interface_depth:interface_depth+5, interface_start:interface_start+interface_width] = interface_reflectivity
        
        return np.clip(profile, 0, 1)
    
    def _apply_depth_attenuation(
        self,
        profile: np.ndarray,
        material_properties: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply depth-dependent attenuation.
        
        Args:
            profile: Reflectivity profile
            material_properties: Material properties
            
        Returns:
            Attenuated profile
        """
        depth, width = profile.shape
        
        # Calculate attenuation coefficient
        scattering_coeff = material_properties['scattering_coefficient'] * 1e-4  # Convert to μm^-1
        absorption_coeff = material_properties['absorption_coefficient'] * 1e-4  # Convert to μm^-1
        total_attenuation = scattering_coeff + absorption_coeff
        
        # Create depth-dependent attenuation
        depth_coords = np.arange(depth)[:, np.newaxis]
        attenuation = np.exp(-total_attenuation * depth_coords * self.axial_resolution)
        
        # Apply attenuation
        attenuated_profile = profile * attenuation
        
        return attenuated_profile
    
    def _generate_a_scan(
        self,
        reflectivity_profile: np.ndarray,
        material_properties: Dict[str, float]
    ) -> np.ndarray:
        """
        Generate OCT A-scan from reflectivity profile.
        
        Args:
            reflectivity_profile: Depth reflectivity profile
            material_properties: Material properties
            
        Returns:
            OCT A-scan
        """
        # Generate complex reflectivity
        complex_reflectivity = reflectivity_profile * np.exp(1j * 2 * np.pi * np.random.random(len(reflectivity_profile)))
        
        # Apply OCT point spread function
        psf = self._create_oct_psf(len(reflectivity_profile))
        
        # Convolve with PSF
        a_scan_complex = np.convolve(complex_reflectivity, psf, mode='same')
        
        # Calculate intensity
        a_scan_intensity = np.abs(a_scan_complex) ** 2
        
        # Apply speckle noise
        speckle_factor = np.random.exponential(1, len(a_scan_intensity))
        a_scan_intensity *= speckle_factor
        
        return a_scan_intensity
    
    def _create_oct_psf(self, length: int) -> np.ndarray:
        """
        Create OCT point spread function.
        
        Args:
            length: Length of PSF
            
        Returns:
            Point spread function
        """
        # Gaussian PSF based on coherence length
        sigma = self.coherence_length / (2 * self.axial_resolution * 1e3)  # Convert to pixels
        x = np.linspace(-length//2, length//2, length)
        psf = np.exp(-(x**2) / (2 * sigma**2))
        
        # Normalize
        psf = psf / np.sum(psf)
        
        return psf
    
    def _apply_internal_crack_oct(
        self,
        oct_image: np.ndarray,
        defect_mask: np.ndarray,
        defect_pixel: int
    ) -> np.ndarray:
        """
        Apply internal crack defect to OCT image.
        
        Args:
            oct_image: OCT image
            defect_mask: Defect mask
            defect_pixel: Defect depth pixel
            
        Returns:
            OCT image with internal crack
        """
        defect_oct = oct_image.copy()
        depth, width = oct_image.shape
        
        # Create crack pattern
        crack_width = 3  # pixels
        crack_start = max(0, defect_pixel - crack_width // 2)
        crack_end = min(depth, defect_pixel + crack_width // 2)
        
        # Apply crack effect
        for x in range(width):
            if defect_mask[x] > 0:
                # Reduce intensity in crack region
                defect_oct[crack_start:crack_end, x] *= 0.3
                
                # Add crack artifacts
                crack_artifact = np.random.normal(0, 0.1, crack_end - crack_start)
                defect_oct[crack_start:crack_end, x] += crack_artifact
        
        return np.clip(defect_oct, 0, 1)
    
    def _apply_material_inclusion_oct(
        self,
        oct_image: np.ndarray,
        defect_mask: np.ndarray,
        defect_pixel: int
    ) -> np.ndarray:
        """
        Apply material inclusion defect to OCT image.
        
        Args:
            oct_image: OCT image
            defect_mask: Defect mask
            defect_pixel: Defect depth pixel
            
        Returns:
            OCT image with material inclusion
        """
        defect_oct = oct_image.copy()
        depth, width = oct_image.shape
        
        # Create inclusion pattern
        inclusion_size = 10  # pixels
        inclusion_start = max(0, defect_pixel - inclusion_size // 2)
        inclusion_end = min(depth, defect_pixel + inclusion_size // 2)
        
        # Apply inclusion effect
        for x in range(width):
            if defect_mask[x] > 0:
                # Different reflectivity for inclusion
                inclusion_reflectivity = 0.2  # Higher reflectivity
                defect_oct[inclusion_start:inclusion_end, x] *= inclusion_reflectivity
                
                # Add inclusion artifacts
                inclusion_artifact = np.random.normal(0, 0.05, inclusion_end - inclusion_start)
                defect_oct[inclusion_start:inclusion_end, x] += inclusion_artifact
        
        return np.clip(defect_oct, 0, 1)
    
    def _apply_delamination_oct(
        self,
        oct_image: np.ndarray,
        defect_mask: np.ndarray,
        defect_pixel: int
    ) -> np.ndarray:
        """
        Apply delamination defect to OCT image.
        
        Args:
            oct_image: OCT image
            defect_mask: Defect mask
            defect_pixel: Defect depth pixel
            
        Returns:
            OCT image with delamination
        """
        defect_oct = oct_image.copy()
        depth, width = oct_image.shape
        
        # Create delamination pattern
        delamination_thickness = 5  # pixels
        delamination_start = max(0, defect_pixel - delamination_thickness // 2)
        delamination_end = min(depth, defect_pixel + delamination_thickness // 2)
        
        # Apply delamination effect
        for x in range(width):
            if defect_mask[x] > 0:
                # High reflectivity at delamination interface
                defect_oct[delamination_start:delamination_end, x] *= 2.0
                
                # Add interface artifacts
                interface_artifact = np.random.normal(0, 0.1, delamination_end - delamination_start)
                defect_oct[delamination_start:delamination_end, x] += interface_artifact
        
        return np.clip(defect_oct, 0, 1)
    
    def _apply_porosity_oct(
        self,
        oct_image: np.ndarray,
        defect_mask: np.ndarray,
        defect_pixel: int
    ) -> np.ndarray:
        """
        Apply porosity defect to OCT image.
        
        Args:
            oct_image: OCT image
            defect_mask: Defect mask
            defect_pixel: Defect depth pixel
            
        Returns:
            OCT image with porosity
        """
        defect_oct = oct_image.copy()
        depth, width = oct_image.shape
        
        # Create porosity pattern
        porosity_region = 15  # pixels
        porosity_start = max(0, defect_pixel - porosity_region // 2)
        porosity_end = min(depth, defect_pixel + porosity_region // 2)
        
        # Apply porosity effect
        for x in range(width):
            if defect_mask[x] > 0:
                # Random porosity pattern
                porosity_pattern = np.random.random(porosity_end - porosity_start)
                porosity_mask = porosity_pattern > 0.7  # 30% porosity
                
                # Reduce intensity in porous regions
                defect_oct[porosity_start:porosity_end, x][porosity_mask] *= 0.5
                
                # Add scattering artifacts
                scatter_artifact = np.random.normal(0, 0.08, porosity_end - porosity_start)
                defect_oct[porosity_start:porosity_end, x] += scatter_artifact
        
        return np.clip(defect_oct, 0, 1)
    
    def _apply_thermal_damage_oct(
        self,
        oct_image: np.ndarray,
        defect_mask: np.ndarray,
        defect_pixel: int
    ) -> np.ndarray:
        """
        Apply thermal damage defect to OCT image.
        
        Args:
            oct_image: OCT image
            defect_mask: Defect mask
            defect_pixel: Defect depth pixel
            
        Returns:
            OCT image with thermal damage
        """
        defect_oct = oct_image.copy()
        depth, width = oct_image.shape
        
        # Create thermal damage pattern
        damage_region = 20  # pixels
        damage_start = max(0, defect_pixel - damage_region // 2)
        damage_end = min(depth, defect_pixel + damage_region // 2)
        
        # Apply thermal damage effect
        for x in range(width):
            if defect_mask[x] > 0:
                # Create thermal damage gradient
                damage_gradient = np.linspace(1.0, 0.3, damage_end - damage_start)
                defect_oct[damage_start:damage_end, x] *= damage_gradient
                
                # Add thermal artifacts
                thermal_artifact = np.random.normal(0, 0.12, damage_end - damage_start)
                defect_oct[damage_start:damage_end, x] += thermal_artifact
        
        return np.clip(defect_oct, 0, 1)
    
    def _apply_shot_noise(self, oct_image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply shot noise to OCT image.
        
        Args:
            oct_image: OCT image
            noise_level: Noise level
            
        Returns:
            Noisy OCT image
        """
        # Shot noise follows Poisson distribution
        photon_counts = oct_image * 1000  # Convert to photon counts
        noisy_counts = np.random.poisson(photon_counts)
        noisy_image = noisy_counts / 1000
        
        return np.clip(noisy_image, 0, 1)
    
    def _apply_speckle_noise(self, oct_image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply speckle noise to OCT image.
        
        Args:
            oct_image: OCT image
            noise_level: Noise level
            
        Returns:
            Noisy OCT image
        """
        # Speckle noise follows exponential distribution
        speckle_factor = np.random.exponential(1, oct_image.shape)
        noisy_image = oct_image * speckle_factor
        
        return np.clip(noisy_image, 0, 1)
    
    def _apply_electronic_noise(self, oct_image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Apply electronic noise to OCT image.
        
        Args:
            oct_image: OCT image
            noise_level: Noise level
            
        Returns:
            Noisy OCT image
        """
        # Electronic noise follows Gaussian distribution
        electronic_noise = np.random.normal(0, noise_level, oct_image.shape)
        noisy_image = oct_image + electronic_noise
        
        return np.clip(noisy_image, 0, 1)
    
    def _apply_motion_artifact(self, oct_image: np.ndarray, artifact_level: float) -> np.ndarray:
        """
        Apply motion artifact to OCT image.
        
        Args:
            oct_image: OCT image
            artifact_level: Artifact level
            
        Returns:
            OCT image with motion artifacts
        """
        depth, width = oct_image.shape
        
        # Create motion pattern
        motion_pattern = np.sin(2 * np.pi * np.arange(width) / width) * artifact_level
        
        # Apply motion artifact
        artifact_image = oct_image.copy()
        for x in range(width):
            shift = int(motion_pattern[x] * depth)
            if shift != 0:
                artifact_image[:, x] = np.roll(oct_image[:, x], shift)
        
        return artifact_image
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate image entropy.
        
        Args:
            image: Input image
            
        Returns:
            Entropy value
        """
        # Normalize image to [0, 255] for histogram
        normalized = (image * 255).astype(np.uint8)
        
        # Calculate histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        
        return entropy


def create_oct_config(
    center_wavelength: float = 1300.0,
    bandwidth: float = 100.0,
    axial_resolution: float = 5.0
) -> Dict[str, Any]:
    """
    Create OCT configuration dictionary.
    
    Args:
        center_wavelength: Center wavelength in nm
        bandwidth: Spectral bandwidth in nm
        axial_resolution: Axial resolution in μm
        
    Returns:
        OCT configuration dictionary
    """
    return {
        'oct_center_wavelength': center_wavelength,
        'oct_bandwidth': bandwidth,
        'oct_axial_resolution': axial_resolution,
        'oct_lateral_resolution': 10.0,
        'oct_depth_range': 2000.0,
        'pixel_size': 1.0
    }