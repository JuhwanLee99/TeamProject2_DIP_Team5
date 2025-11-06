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
    (Needs Implementation)
    Plots the R, G, B histograms for both original and corrected images.
    """
    print("Histogram plotting (Needs Implementation)")
    
    # (Implementation-Hint: Use matplotlib.pyplot.hist)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ...
    # for i, color in enumerate(['r', 'g', 'b']):
    #     ax1.hist(img_rgb[..., i].ravel(), bins=256, color=color, alpha=0.7)
    # ax1.set_title("Original Histogram")
    # ... (repeat for ax2 with corrected_rgb)
    # plt.show()
    pass