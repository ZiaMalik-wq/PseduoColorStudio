import cv2
import numpy as np


def apply_sin_mapping(gray_img: np.ndarray, frequency: float = 1.0) -> np.ndarray:
    """
    Map intensity to color using sine functions with phase offsets per channel.
    Produces smooth, cyclic rainbow-like colorization.

    R = sin(f * pi * I)
    G = sin(f * pi * I + 2pi/3)
    B = sin(f * pi * I + 4pi/3)
    """
    gray = _prepare_gray(gray_img)
    f = frequency * np.pi

    gray_vals = np.arange(256, dtype=np.float32) / 255.0
    R = np.sin(f * gray_vals)
    G = np.sin(f * gray_vals + 2 * np.pi / 3)
    B = np.sin(f * gray_vals + 4 * np.pi / 3)

    R = ((R + 1) / 2 * 255).astype(np.uint8)
    G = ((G + 1) / 2 * 255).astype(np.uint8)
    B = ((B + 1) / 2 * 255).astype(np.uint8)

    lut = np.stack([B, G, R], axis=-1).reshape(256, 1, 3)
    colored = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)
    return colored


def apply_histogram_equalized_lut(gray_img: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Apply histogram equalization before colormap mapping."""
    import algorithms.lut as lut_module

    gray = _prepare_gray(gray_img)
    equalized = cv2.equalizeHist(gray)
    return lut_module.apply_lut(equalized, colormap)


def apply_gamma_mapped(
    gray_img: np.ndarray,
    gamma: float = 1.5,
    colormap: str = "plasma",
) -> np.ndarray:
    """Apply gamma correction then colormap."""
    import algorithms.lut as lut_module

    gray = _prepare_gray(gray_img)

    gray_vals = np.arange(256, dtype=np.float32) / 255.0
    corrected = np.power(gray_vals, 1.0 / gamma)
    lut = (corrected * 255).astype(np.uint8)

    gray_corrected = cv2.LUT(gray, lut)
    return lut_module.apply_lut(gray_corrected, colormap)


def apply_density_mapping(gray_img: np.ndarray) -> np.ndarray:
    """Density-style coloring: maps intensity to a blue-green-yellow-red scale."""
    gray = _prepare_gray(gray_img)

    gray_vals = np.arange(256, dtype=np.float32) / 255.0
    R = np.where(gray_vals < 0.5, 0.0, (gray_vals - 0.5) * 2)
    G = np.where(gray_vals < 0.5, gray_vals * 2, 1.0 - (gray_vals - 0.5) * 2)
    B = np.where(gray_vals < 0.5, 1.0 - gray_vals * 2, 0.0)

    R = (np.clip(R, 0, 1) * 255).astype(np.uint8)
    G = (np.clip(G, 0, 1) * 255).astype(np.uint8)
    B = (np.clip(B, 0, 1) * 255).astype(np.uint8)

    lut = np.stack([B, G, R], axis=-1).reshape(256, 1, 3)
    colored = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)
    return colored


def _prepare_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) \
              if img.max() <= 1.0 \
              else np.clip(img, 0, 255).astype(np.uint8)
    return img
