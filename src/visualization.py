#src/visualization.py

import matplotlib.pyplot as plt
import numpy as np

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

def plot_histograms(img_rgb, corrected_rgb):
    """
    Plots the R, G, B histograms for both original and corrected images side-by-side.
    """
    # Ensure images are within valid display range
    img_rgb = np.clip(img_rgb, 0, 255)
    corrected_rgb = np.clip(corrected_rgb, 0, 255)

    colors = ['r', 'g', 'b']
    bins = np.arange(257)  # 256 bins for 0-255 range

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for idx, color in enumerate(colors):
        hist, bin_edges = np.histogram(img_rgb[..., idx].ravel(), bins=bins, range=(0, 255))
        ax1.plot(bin_edges[:-1], hist, color=color, label=f"{color.upper()} channel")

        corrected_hist, corrected_edges = np.histogram(corrected_rgb[..., idx].ravel(), bins=bins, range=(0, 255))
        ax2.plot(corrected_edges[:-1], corrected_hist, color=color, label=f"{color.upper()} channel")

    ax1.set_title("Original Image Histograms")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    ax2.set_title("Corrected Image Histograms")
    ax2.set_xlabel("Pixel Intensity")
    ax2.legend()

    plt.tight_layout()
    plt.show(block=True)