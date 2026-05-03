import logging
import cv2
import numpy as np

from algorithms.utils import prepare_gray

# Available colormaps (OpenCV built-ins + custom)
COLORMAPS = {
    "jet":      cv2.COLORMAP_JET,
    "hot":      cv2.COLORMAP_HOT,
    "cool":     cv2.COLORMAP_COOL,
    "hsv":      cv2.COLORMAP_HSV,
    "rainbow":  cv2.COLORMAP_RAINBOW,
    "magma":    cv2.COLORMAP_MAGMA,
    "inferno":  cv2.COLORMAP_INFERNO,
    "plasma":   cv2.COLORMAP_PLASMA,
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "turbo":    cv2.COLORMAP_TURBO,
}

logger = logging.getLogger(__name__)


def apply_lut(gray_img: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """
    Apply a colormap LUT to a grayscale image.

    Args:
        gray_img:  Grayscale image, shape (H, W) or (H, W, 1), uint8 or float
        colormap:  Name from COLORMAPS dict

    Returns:
        colored:   BGR image (H, W, 3) uint8
    """
    gray = prepare_gray(gray_img)

    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choose from: {list(COLORMAPS.keys())}")

    colored = cv2.applyColorMap(gray, COLORMAPS[colormap])
    return colored


def apply_custom_lut(gray_img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply a user-defined 256-entry LUT.

    Args:
        gray_img:  Grayscale image
        lut:       numpy array of shape (256, 1, 3) — BGR values for each intensity

    Returns:
        colored:   BGR image (H, W, 3) uint8
    """
    gray = prepare_gray(gray_img)
    colored = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)
    return colored


def build_custom_lut(
    low_color:  tuple = (255, 0,   0),
    mid_color:  tuple = (0,   255, 0),
    high_color: tuple = (0,   0,   255),
) -> np.ndarray:
    """Build a smooth 3-stop gradient LUT from three BGR anchor colors."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    t = np.linspace(0, 1, 128)[:, None]

    low = np.array(low_color)
    mid = np.array(mid_color)
    high = np.array(high_color)

    lut[:128, 0, :] = (low * (1 - t) + mid * t).astype(np.uint8)
    lut[128:, 0, :] = (mid * (1 - t) + high * t).astype(np.uint8)

    return lut



