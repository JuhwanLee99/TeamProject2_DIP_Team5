# src/analysis.py

from typing import List, Tuple

import numpy as np


def compute_channel_histogram(channel: np.ndarray, bins: int = 256, value_range: Tuple[float, float] = (0, 255)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a histogram for a single channel.
    """
    hist, bin_edges = np.histogram(channel.ravel(), bins=bins, range=value_range)
    return hist, bin_edges


def compute_rgb_histograms(
    img_rgb: np.ndarray,
    bins: int = 256,
    value_range: Tuple[float, float] = (0, 255),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel histograms for an RGB image.
    """
    if img_rgb.ndim < 3 or img_rgb.shape[-1] < 3:
        raise ValueError("img_rgb must have at least three channels in the last dimension")

    histograms: List[np.ndarray] = []
    bin_edges = None

    for channel_idx in range(3):
        hist, bin_edges = compute_channel_histogram(img_rgb[..., channel_idx], bins=bins, value_range=value_range)
        histograms.append(hist)

    return np.stack(histograms, axis=0), bin_edges