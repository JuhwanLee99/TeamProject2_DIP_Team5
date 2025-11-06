#src/io_utils.py

import cv2
import numpy as np

def read_image(path):
    """
    Reads an image using OpenCV.
    Returns:
        NumPy array (H, W, 3) in BGR format, or None if failed.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read image. Check path: {path}")
    return img

def save_image(path, img_bgr):
    """
    Saves a NumPy array (BGR format) as an image.
    """
    # Ensure image is in 0-255 uint8 range
    img_to_save = np.clip(img_bgr, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_to_save)