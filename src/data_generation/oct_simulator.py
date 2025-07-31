import numpy as np
from typing import Tuple


def simulate_oct(mask: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Generate a synthetic OCT B-scan based on a binary defect mask."""
    # Basic: random speckle + intensity fall-off + mask-dependent signal drop
    h, w = img_size
    base = np.random.normal(0.5, 0.1, (h, w)).astype(np.float32)
    # Depth-dependent attenuation
    depth_vec = np.linspace(1.0, 0.4, h).reshape(-1, 1)
    base *= depth_vec
    # Introduce lower reflectance inside defect regions
    attenuated = base * (1.0 - 0.5 * mask)
    # Normalise
    attenuated = (attenuated - attenuated.min()) / (attenuated.max() - attenuated.min())
    return (attenuated * 255).astype(np.uint8)