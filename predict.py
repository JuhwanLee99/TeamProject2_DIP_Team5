#predict.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.ai_scene import classify_image
from src.ai_optimize import (
    generate_preview,
    load_diagnostic_model,
    optimize_corrections,
)
from src.correction import apply_all_corrections_torch
from src.io_utils import read_image, save_image
from src.visualization import plot_with_filter_gallery


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


def _apply_params(img_rgb: np.ndarray, params: dict, device: torch.device) -> np.ndarray:
    """
    Convenience helper to run the manual correction pipeline for an arbitrary
    parameter dictionary.
    """
    input_tensor = preprocess(img_rgb, device)

    params_tensor = torch.tensor(
        [
            [
                params["gamma"],
                params["gain_r"],
                params["gain_g"],
                params["gain_b"],
                params["sat"],
                params["hue"],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    corrected_tensor = apply_all_corrections_torch(input_tensor, params_tensor)
    return postprocess(corrected_tensor[0])


def _validate_application(
    img_rgb: np.ndarray, advice, device: torch.device
) -> tuple[np.ndarray, float, int, float]:
    """
    Confirm the preview path and manual parameter application match and report
    how much the image changed from the original.
    """

    # Preview path (uses advice.to_tensor internally)
    preview_image_np = generate_preview(img_rgb, advice, device=device)

    # Manual application path
    validated_image_np = _apply_params(img_rgb, advice.final_params, device)

    # Compare both correction paths to ensure parameters are applied consistently
    diff = np.abs(preview_image_np.astype(np.int16) - validated_image_np.astype(np.int16))
    mean_diff = float(diff.mean())
    max_diff = int(diff.max())

    # Measure the actual impact of the parameters relative to the original
    impact = np.abs(validated_image_np.astype(np.int16) - img_rgb.astype(np.int16))
    mean_impact = float(impact.mean())

    return validated_image_np, mean_diff, max_diff, mean_impact


def predict(
    weights_path: str | Path,
    input_path: str | Path,
    fusion_weight: float = 0.6,
    diagnostic_model_path: str | Path | None = None,
    save_presets: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading and preprocessing image from {input_path}...")
    img_bgr = read_image(input_path)
    if img_bgr is None:
        return

    # Ensure a contiguous RGB array; slicing with ::-1 can create negative strides
    # that PyTorch tensors do not support.
    img_rgb = np.ascontiguousarray(img_bgr[..., ::-1])  # BGR -> RGB

    print("Classifying scene with MobileNetV2...")
    scene_info = classify_image(img_rgb, weights_path=weights_path)
    print(
        "Predicted scene: {scene} (confidence: {score:.3f}, raw label: {raw})".format(
            scene=scene_info["scene"], score=scene_info["score"], raw=scene_info["raw_label"]
        )
    )

    diagnostic_model = None
    if diagnostic_model_path is not None:
        diagnostic_model = load_diagnostic_model(Path(diagnostic_model_path), device=device)

    print("Running AI diagnostic optimizer (exposure, white balance, saturation)...")
    advice = optimize_corrections(
        img_rgb,
        scene_info,
        diagnostic_model=diagnostic_model,
        device=device,
        fusion_weight=fusion_weight,
    )

    advice_text = str(advice)
    print("\n" + advice_text + "\n")

    print("Generating corrected preview with fused parameters...")
    print("Validating applied parameters via manual correction pipeline...")
    corrected_image_np, mean_diff, max_diff, mean_impact = _validate_application(
        img_rgb, advice, device
    )

    if max_diff > 2 or mean_diff > 0.5:
        print(
            f"Warning: Preview path mismatch detected (mean diff {mean_diff:.2f}, max {max_diff}). "
            "Using validated pipeline output for display and export."
        )
    else:
        print(
            f"Validated parameter application (mean diff {mean_diff:.2f}, max {max_diff}). "
            "Using validated pipeline output."
        )

    print(
        f"Applied parameters visibly changed the image (mean abs pixel shift vs. original: {mean_impact:.2f})."
    )

    # Build gallery of previews (fused + recommended filters)
    filter_previews: list[dict] = [{"name": "AI + Preset", "image": corrected_image_np}]

    recommended_previews: list[dict] = []
    if advice.recommended_filters:
        for preset in advice.recommended_filters:
            preset_img = _apply_params(img_rgb, preset["params"], device)
            recommended_previews.append({"name": preset["name"], "image": preset_img})

        filter_previews.extend(recommended_previews)

    print("Displaying results...")
    info_text = (
        f"Scene: {scene_info['scene']} (confidence {scene_info['score']:.2f}, raw={scene_info['raw_label']})\n\n"
        "Final Parameters:\n"
        f"  gamma={advice.final_params['gamma']:.3f}\n"
        f"  gains: R={advice.final_params['gain_r']:.3f}, "
        f"G={advice.final_params['gain_g']:.3f}, B={advice.final_params['gain_b']:.3f}\n"
        f"  saturation={advice.final_params['sat']:.3f}, hue={advice.final_params['hue']:.3f}\n\n"
        "Adjustments:\n"
        f"  Exposure: {advice.exposure_adjustment_percent:+.1f}%\n"
        f"  White Balance: {advice.white_balance_shift_percent:.1f}% shift\n"
        f"  Saturation: {advice.saturation_adjustment_percent:+.1f}%\n\n"
        "Summary:\n  " + "\n  ".join(advice.corrections_summary)
    )

    plot_with_filter_gallery(
        img_rgb,
        filter_previews,
        info_text=info_text,
        base_corrected_title="Corrected",
    )

    output_path = Path("data/output/scene_corrected.jpg")
    save_image(output_path, corrected_image_np[..., ::-1])  # Convert back to BGR
    print(f"Corrected image saved to {output_path}")

    if save_presets and recommended_previews:
        print("Saving additional recommended filter presets...")
        presets_dir = Path("data/output/presets")
        presets_dir.mkdir(parents=True, exist_ok=True)

        for preset in recommended_previews:
            preset_path = presets_dir / f"{preset['name'].lower()}_preview.jpg"
            save_image(preset_path, preset["image"][..., ::-1])
            print(f" - {preset['name']} saved to {preset_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene-based color correction with AI optimization.")
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the MobileNetV2 weights (.pth) for scene classification.",
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--fusion_weight",
        type=float,
        default=0.6,
        help="Blend ratio between AI diagnostics and scene preset (1.0 = AI only, 0.0 = preset only).",
    )
    parser.add_argument(
        "--diagnostic_model_path",
        type=str,
        default=None,
        help="Optional path to the diagnostic CNN weights (.pth). Uses bundled model if omitted.",
    )
    parser.add_argument(
        "--disable_presets",
        action="store_true",
        help="Skip saving the recommended filter preset previews.",
    )
    args = parser.parse_args()

    predict(
        args.weights_path,
        args.input_path,
        fusion_weight=args.fusion_weight,
        diagnostic_model_path=args.diagnostic_model_path,
        save_presets=not args.disable_presets,
    )