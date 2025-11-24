#src/visualization.py

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import compute_rgb_histograms

def plot_side_by_side(img1, img2, title1="Original", title2="Corrected"):
    """
    Displays two NumPy images (RGB format) side-by-side.
    """
    # Ensure images are in displayable format (0-255, uint8)
    img1_disp = np.clip(img1, 0, 255).astype(np.uint8)
    img2_disp = np.clip(img2, 0, 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(img1_disp)
    ax1.set_title(title1)
    ax1.axis('off')

    ax2.imshow(img2_disp)
    ax2.set_title(title2)
    ax2.axis('off')

    plt.tight_layout()
    plt.show(block=True) # Use block=True to ensure window stays open

def plot_histograms(img_rgb, corrected_rgb, bins: int = 256):
    """
    Plots the R, G, B histograms for both original and corrected images side-by-side.
    """
    # Ensure images are within valid display range
    original_hists, bin_edges = compute_rgb_histograms(img_rgb, bins=bins)
    corrected_hists, _ = compute_rgb_histograms(corrected_rgb, bins=bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    bin_width = np.diff(bin_edges)[0]
    bin_positions = bin_edges[:-1]

    for channel_idx, color in enumerate(['r', 'g', 'b']):
        ax1.bar(bin_positions, original_hists[channel_idx], color=color, alpha=0.6, width=bin_width)
        ax2.bar(bin_positions, corrected_hists[channel_idx], color=color, alpha=0.6, width=bin_width)

    ax1.set_title("Original Histogram")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(bin_edges[0], bin_edges[-1])

    ax2.set_title("Corrected Histogram")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(bin_edges[0], bin_edges[-1])

    plt.tight_layout()
    plt.show(block=True)