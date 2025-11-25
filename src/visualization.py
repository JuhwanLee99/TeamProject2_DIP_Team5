#src/visualization.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from src.analysis import compute_rgb_histograms


def plot_side_by_side(img1, img2, title1="Original", title2="Corrected", info_text: str | None = None):
    """
    Displays two NumPy images (RGB format) side-by-side.

    Args:
        img1: Original image (RGB, HxWx3).
        img2: Corrected image (RGB, HxWx3).
        title1: Title for the first image panel.
        title2: Title for the second image panel.
        info_text: Optional textual metadata (scene + parameters) to show
            alongside the images.
    """
    # Ensure images are in displayable format (0-255, uint8)
    img1_disp = np.clip(img1, 0, 255).astype(np.uint8)
    img2_disp = np.clip(img2, 0, 255).astype(np.uint8)

    ncols = 3 if info_text else 2
    figsize = (16, 6) if info_text else (12, 6)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax1 = axes[0]
    ax1.imshow(img1_disp)
    ax1.set_title(title1)
    ax1.axis("off")

    ax2 = axes[1]
    ax2.imshow(img2_disp)
    ax2.set_title(title2)
    ax2.axis("off")

    if info_text:
        ax3 = axes[2]
        ax3.axis("off")
        ax3.text(
            0.0,
            1.0,
            info_text,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            family="monospace",
            wrap=True,
        )

    plt.tight_layout()
    plt.show(block=True)  # Use block=True to ensure window stays open


def plot_with_filter_gallery(
    original_img: np.ndarray,
    filter_previews: list[dict],
    info_text: str | None = None,
    base_corrected_title: str = "Corrected",
):
    """
    Display original + corrected images with a clickable gallery of filter previews.

    Args:
        original_img: Original RGB image (HxWx3, uint8 or float).
        filter_previews: List of {"name": str, "image": np.ndarray (RGB)} items. The
            first entry is treated as the default corrected image.
        info_text: Optional text to display beside the images.
        base_corrected_title: Title for the default corrected image selection.
    """

    if not filter_previews:
        raise ValueError("filter_previews must contain at least one image")

    # Prepare displayable copies
    original_disp = np.clip(original_img, 0, 255).astype(np.uint8)
    previews_disp = [
        {"name": item["name"], "image": np.clip(item["image"], 0, 255).astype(np.uint8)}
        for item in filter_previews
    ]

    has_info = info_text is not None
    main_cols = 3 if has_info else 2

    fig = plt.figure(figsize=(16, 10) if has_info else (14, 9))
    gs = GridSpec(2, main_cols, height_ratios=[4, 1.3], figure=fig)

    # Main image axes
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[0, 2]) if has_info else None

    ax_orig.imshow(original_disp)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    corrected_im = ax_corr.imshow(previews_disp[0]["image"])
    ax_corr.set_title(f"{base_corrected_title} ({previews_disp[0]['name']})")
    ax_corr.axis("off")

    if has_info:
        ax_info.axis("off")
        ax_info.text(
            0.0,
            1.0,
            info_text,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            family="monospace",
            wrap=True,
        )

    # Thumbnail gallery along the bottom spanning all columns
    thumb_gs = gs[1, :].subgridspec(1, len(previews_disp), wspace=0.1)
    thumb_axes = []
    for idx, preview in enumerate(previews_disp):
        ax_thumb = fig.add_subplot(thumb_gs[0, idx])
        ax_thumb.imshow(preview["image"])
        ax_thumb.set_title(preview["name"], fontsize=9)
        ax_thumb.axis("off")
        thumb_axes.append(ax_thumb)

    # Selection highlight
    selected_idx = 0
    thumb_axes[selected_idx].set_title(f"{previews_disp[0]['name']} ✓", fontsize=9, color="tab:blue")

    def on_click(event):
        nonlocal selected_idx
        if event.inaxes in thumb_axes:
            new_idx = thumb_axes.index(event.inaxes)
            if new_idx == selected_idx:
                return

            # Update highlight
            thumb_axes[selected_idx].set_title(previews_disp[selected_idx]["name"], fontsize=9, color="black")
            thumb_axes[new_idx].set_title(f"{previews_disp[new_idx]['name']} ✓", fontsize=9, color="tab:blue")
            selected_idx = new_idx

            # Update corrected image
            corrected_im.set_data(previews_disp[new_idx]["image"])
            ax_corr.set_title(f"{base_corrected_title} ({previews_disp[new_idx]['name']})")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show(block=True)

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