#predict.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.ai_scene import classify_image
from src.correction import apply_all_corrections_torch
from src.io_utils import read_image, save_image
from src.visualization import plot_side_by_side


# --- Pre/Post-processing helpers ---
def preprocess(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Apply preprocessing before running the manual correction pipeline.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (H,W,C) 0-255 -> (C,H,W) 0-1
        ]
    )

    tensor = transform(image_np.copy()).to(device)
    return tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)


def postprocess(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor back to a displayable NumPy image.
    """
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
    return image_np


def predict(weights_path: str | Path, input_path: str | Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading and preprocessing image from {input_path}...")
    img_bgr = read_image(input_path)
    if img_bgr is None:
        return

    img_rgb = img_bgr[..., ::-1]  # BGR -> RGB

    print("Classifying scene with MobileNetV2...")
    scene_info = classify_image(img_rgb, weights_path=weights_path)
    scene_params = scene_info["scene_corrections"]
    print(
        "Predicted scene: {scene} (confidence: {score:.3f}, raw label: {raw})".format(
            scene=scene_info["scene"], score=scene_info["score"], raw=scene_info["raw_label"]
        )
    )

    params_tensor = torch.tensor(
        [
            [
                scene_params["gamma"],
                scene_params["white_balance_r"],
                scene_params["white_balance_g"],
                scene_params["white_balance_b"],
                scene_params["saturation"],
                scene_params["hue_shift"],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    input_tensor = preprocess(img_rgb, device)

    print("Applying manual correction functions...")
    corrected_tensor = apply_all_corrections_torch(input_tensor, params_tensor)

    corrected_image_np = postprocess(corrected_tensor[0])

    print("Displaying results...")
    plot_side_by_side(img_rgb, corrected_image_np, "Original", "Corrected (Preset)")

    output_path = Path("data/output/scene_corrected.jpg")
    save_image(output_path, corrected_image_np[..., ::-1])  # Convert back to BGR
    print(f"Corrected image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene-based color correction.")
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the MobileNetV2 weights (.pth) for scene classification.",
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    predict(args.weights_path, args.input_path)