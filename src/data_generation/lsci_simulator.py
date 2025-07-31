import numpy as np
import cv2
from typing import Tuple


def simulate_lsci(mask: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Generate a synthetic Laser Speckle Contrast image."""
    h, w = img_size
    # Create speckle by convolving white noise with a Gaussian kernel.
    noise = np.random.rand(h, w).astype(np.float32)
    kernel_size = 9
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, 0)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
    speckle = cv2.filter2D(noise, -1, gaussian_kernel)
    speckle = (speckle - speckle.min()) / (speckle.max() - speckle.min())
    # Higher flow (non-defect) ⇒ lower contrast ⇒ brighter pixels
    contrast_map = 1 - 0.5 * mask  # defect region darker
    img = speckle * contrast_map
    return (img * 255).astype(np.uint8)