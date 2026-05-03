import cv2
import numpy as np


def prepare_gray(img: np.ndarray, *, bgr_to_gray: bool = False) -> np.ndarray:
    """Ensure *img* is a single-channel uint8 grayscale array.

    Parameters
    ----------
    img : np.ndarray
        Input image — may be grayscale (H, W), single-channel (H, W, 1),
        or 3-channel BGR (H, W, 3).  Dtype may be uint8 or float.
    bgr_to_gray : bool, default False
        If *True*, 3-channel images are converted with the proper
        ``cv2.cvtColor(COLOR_BGR2GRAY)`` weighted formula.
        If *False*, only the first channel is taken (faster, fine when
        the input is already grayscale stored as 3-channel).

    Returns
    -------
    np.ndarray
        Grayscale image, shape (H, W), dtype uint8.
    """
    if img.ndim == 3:
        if bgr_to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]

    if img.dtype != np.uint8:
        img = (
            (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if img.max() <= 1.0
            else np.clip(img, 0, 255).astype(np.uint8)
        )

    return img
