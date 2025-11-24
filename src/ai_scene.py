# src/ai_scene.py

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# Preset correction parameters keyed by scene label.
SCENE_CORRECTIONS: Dict[str, Dict[str, float]] = {
    "Landscape": {
        "gamma": 1.05,
        "white_balance_r": 1.05,
        "white_balance_g": 1.00,
        "white_balance_b": 0.98,
        "saturation": 1.08,
        "hue_shift": 0.00,
    },
    "Portrait": {
        "gamma": 0.95,
        "white_balance_r": 1.02,
        "white_balance_g": 1.00,
        "white_balance_b": 1.03,
        "saturation": 1.02,
        "hue_shift": 0.01,
    },
    "Food": {
        "gamma": 1.00,
        "white_balance_r": 1.08,
        "white_balance_g": 1.00,
        "white_balance_b": 0.96,
        "saturation": 1.12,
        "hue_shift": 0.00,
    },
    "Animal": {
        "gamma": 1.00,
        "white_balance_r": 1.02,
        "white_balance_g": 1.00,
        "white_balance_b": 1.00,
        "saturation": 1.05,
        "hue_shift": 0.00,
    },
    "Urban": {
        "gamma": 1.03,
        "white_balance_r": 1.00,
        "white_balance_g": 1.00,
        "white_balance_b": 1.02,
        "saturation": 1.04,
        "hue_shift": 0.00,
    },
    "Vehicle": {
        "gamma": 1.00,
        "white_balance_r": 1.00,
        "white_balance_g": 1.00,
        "white_balance_b": 1.00,
        "saturation": 1.00,
        "hue_shift": 0.00,
    },
    "Plant": {
        "gamma": 1.02,
        "white_balance_r": 1.00,
        "white_balance_g": 1.00,
        "white_balance_b": 0.98,
        "saturation": 1.10,
        "hue_shift": 0.00,
    },
    "Generic": {
        "gamma": 1.00,
        "white_balance_r": 1.00,
        "white_balance_g": 1.00,
        "white_balance_b": 1.00,
        "saturation": 1.00,
        "hue_shift": 0.00,
    },
}

_DEFAULT_WEIGHTS = models.MobileNet_V2_Weights.IMAGENET1K_V1
_WEIGHT_ENV_VAR = "AI_SCENE_MOBILENET_PATH"
_WEIGHT_FILENAME = _DEFAULT_WEIGHTS.url.rsplit("/", 1)[-1]

_model: torch.nn.Module | None = None
_preprocess: transforms.Compose | None = None
_imagenet_labels: Tuple[str, ...] | None = None


def _resolve_weights_path(explicit_path: Optional[str]) -> Path:
    """
    Resolve a local checkpoint path without triggering network downloads.

    """

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Provided weights_path does not exist: {candidate}"
            )
        return candidate

    env_path = os.environ.get(_WEIGHT_ENV_VAR)
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Environment variable {_WEIGHT_ENV_VAR} points to missing weights: {candidate}"
            )
        return candidate

    checkpoints_dir = Path(torch.hub._get_torch_home()) / "checkpoints"
    cached_path = checkpoints_dir / _WEIGHT_FILENAME
    if cached_path.is_file():
        return cached_path

    raise RuntimeError(
        "MobileNetV2 weights not found locally. Provide a checkpoint via "
        f"`classify_image(..., weights_path=...)` or set ${_WEIGHT_ENV_VAR}. "
        f"Expected filename: {_WEIGHT_FILENAME}"
    )


def _load_classifier(weights_path: Optional[str] = None) -> None:
    """Lazy-loads the MobileNetV2 classifier and preprocessing pipeline."""

    global _model, _preprocess, _imagenet_labels
    if _model is not None:
        return

    resolved_path = _resolve_weights_path(weights_path)
    state_dict = torch.load(resolved_path, map_location="cpu")

    _model = models.mobilenet_v2(weights=None)
    _model.load_state_dict(state_dict)
    _model.eval()

    _preprocess = _DEFAULT_WEIGHTS.transforms()
    _imagenet_labels = tuple(_DEFAULT_WEIGHTS.meta["categories"])


def _map_label_to_scene(predicted_label: str) -> str:

    label = predicted_label.lower()
    if any(
        keyword in label
        for keyword in (
            "valley",
            "mountain",
            "volcano",
            "cliff",
            "lakeside",
            "seashore",
            "promontory",
            "sandbar",
            "iceberg",
            "forest",
            "desert",
            "geyser",
        )
    ):
        return "Landscape"
    if any(keyword in label for keyword in ("person", "woman", "man", "boy", "girl", "bride", "groom", "face", "mask")):
        return "Portrait"
    if any(
        keyword in label
        for keyword in (
            "pizza",
            "plate",
            "sandwich",
            "burger",
            "hotdog",
            "spaghetti",
            "soup",
            "salad",
            "coffee",
            "espresso",
            "wine",
            "beer",
            "cocktail",
        )
    ):
        return "Food"
    if any(
        keyword in label
        for keyword in (
            "dog",
            "cat",
            "bird",
            "animal",
            "horse",
            "tiger",
            "lion",
            "bear",
            "wolf",
            "zebra",
        )
    ):
        return "Animal"
    if any(
        keyword in label
        for keyword in (
            "building",
            "skyscraper",
            "street",
            "downtown",
            "mosque",
            "church",
            "tower",
            "castle",
            "library",
            "palace",
            "monastery",
            "plaza",
            "bridge",
        )
    ):
        return "Urban"
    if any(
        keyword in label
        for keyword in (
            "car",
            "bus",
            "truck",
            "van",
            "bicycle",
            "motorcycle",
            "train",
            "airliner",
            "airplane",
            "boat",
            "ship",
            "subway",
        )
    ):
        return "Vehicle"
    if any(
        keyword in label
        for keyword in (
            "flower",
            "rose",
            "tulip",
            "sunflower",
            "lotus",
            "water lily",
            "orchid",
            "daisy",
            "leaf",
        )
    ):
        return "Plant"
    return "Generic"


def classify_image(img_rgb: np.ndarray, weights_path: Optional[str] = None) -> Dict[str, object]:
    """
    Classify an RGB image into a coarse scene label and expose correction hints.
    """
    _load_classifier(weights_path)
    assert _model is not None
    assert _preprocess is not None
    assert _imagenet_labels is not None

    pil_img = Image.fromarray(np.uint8(img_rgb))
    input_tensor = _preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        score, idx = probs.max(dim=1)

    imagenet_label = _imagenet_labels[int(idx.item())]
    scene_label = _map_label_to_scene(imagenet_label)
    corrections = SCENE_CORRECTIONS.get(scene_label, SCENE_CORRECTIONS["Generic"])

    return {
        "scene": scene_label,
        "score": float(score.item()),
        "raw_label": imagenet_label,
        "corrections": corrections,
    }