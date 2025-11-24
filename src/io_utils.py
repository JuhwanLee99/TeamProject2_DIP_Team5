#src/io_utils.py

import cv2
import numpy as np
from pathlib import Path


def read_image(path: str | Path):
    """
    Read an image using OpenCV.
    """
    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: Could not read image. Check path: {path}")
    return img


def save_image(path: str | Path, img_bgr: np.ndarray) -> None:
    """Save a NumPy image (BGR format) to disk, creating directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure image is in 0-255 uint8 range
    img_to_save = np.clip(img_bgr, 0, 255).astype(np.uint8)
    success = cv2.imwrite(str(output_path), img_to_save)
    if not success:
        print(f"Error: Failed to write image to {output_path}")