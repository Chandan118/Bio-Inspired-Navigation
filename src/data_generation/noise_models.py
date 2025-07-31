import numpy as np
import cv2


def add_gaussian_noise(image: np.ndarray, std: float = 0.05):
    noisy = image.astype(np.float32) / 255.0
    noisy += np.random.normal(0, std, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)


def add_motion_blur(image: np.ndarray, kernel_size: int = 7):
    # Create horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)