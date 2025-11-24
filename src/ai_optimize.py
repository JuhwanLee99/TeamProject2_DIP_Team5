# src/ai_optimize.py

"""
AI-powered color correction optimization module.

Upstream: ai_scene.py (provides scene_result dict)
Downstream: correction.py (receives CorrectionAdvice)

Uses CNN to diagnose exposure, white balance, and saturation issues.
Fuses AI diagnostics with scene presets to generate correction parameters.
Does NOT apply corrections - only provides parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models, transforms


# ============================================================================
# DATA STRUCTURES - Output Contracts (CANNOT CHANGE)
# ============================================================================

@dataclass
class ExposureDiagnostic:
    """AI-predicted exposure analysis results."""
    underexposed_probability: float
    overexposed_probability: float
    well_exposed_probability: float
    suggested_gamma: float
    confidence: float
    
    def __str__(self) -> str:
        return (f"Exposure: under={self.underexposed_probability:.2f}, "
                f"over={self.overexposed_probability:.2f}, "
                f"well={self.well_exposed_probability:.2f} "
                f"→ gamma={self.suggested_gamma:.2f}")


@dataclass
class WhiteBalanceDiagnostic:
    """AI-predicted white balance analysis results."""
    color_temperature_kelvin: float
    tint_shift: float
    suggested_gains: Tuple[float, float, float]
    confidence: float
    
    def __str__(self) -> str:
        r, g, b = self.suggested_gains
        return (f"WhiteBalance: {self.color_temperature_kelvin:.0f}K, "
                f"tint={self.tint_shift:+.2f} "
                f"→ gains=({r:.3f}, {g:.3f}, {b:.3f})")


@dataclass
class SaturationDiagnostic:
    """AI-predicted saturation analysis results."""
    undersaturated_probability: float
    oversaturated_probability: float
    current_saturation_level: float
    suggested_adjustment: float
    confidence: float
    
    def __str__(self) -> str:
        return (f"Saturation: under={self.undersaturated_probability:.2f}, "
                f"over={self.oversaturated_probability:.2f}, "
                f"level={self.current_saturation_level:.2f} "
                f"→ adjust={self.suggested_adjustment:.2f}")


@dataclass
class CorrectionAdvice:
    """
    Complete correction advice package for downstream correction.py.
    PRIMARY OUTPUT - INTERFACE CANNOT CHANGE.
    """
    scene_label: str
    scene_confidence: float
    
    exposure: ExposureDiagnostic
    white_balance: WhiteBalanceDiagnostic
    saturation: SaturationDiagnostic
    
    final_params: Dict[str, float]  # {gamma, gain_r, gain_g, gain_b, sat, hue}
    
    corrections_summary: List[str] = field(default_factory=list)
    exposure_adjustment_percent: float = 0.0
    white_balance_shift_percent: float = 0.0
    saturation_adjustment_percent: float = 0.0
    recommended_filters: List[Dict[str, float]] = field(default_factory=list)
    
    def to_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        """Convert final_params to tensor for apply_all_corrections_torch."""
        if device is None:
            device = torch.device('cpu')
        
        params = torch.tensor([[
            self.final_params['gamma'],
            self.final_params['gain_r'],
            self.final_params['gain_g'],
            self.final_params['gain_b'],
            self.final_params['sat'],
            self.final_params['hue']
        ]], dtype=torch.float32, device=device)
        
        return params
    
    def __str__(self) -> str:
        lines = [
            f"=== CorrectionAdvice for '{self.scene_label}' (confidence={self.scene_confidence:.2f}) ===",
            "",
            str(self.exposure),
            str(self.white_balance),
            str(self.saturation),
            "",
            "Final Parameters:",
            f"  gamma: {self.final_params['gamma']:.3f}",
            f"  gains: R={self.final_params['gain_r']:.3f}, "
            f"G={self.final_params['gain_g']:.3f}, B={self.final_params['gain_b']:.3f}",
            f"  saturation: {self.final_params['sat']:.3f}",
            f"  hue_shift: {self.final_params['hue']:.3f}",
            "",
            "Adjustments:",
            f"  Exposure: {self.exposure_adjustment_percent:+.1f}%",
            f"  White Balance: {self.white_balance_shift_percent:.1f}% shift",
            f"  Saturation: {self.saturation_adjustment_percent:+.1f}%",
            "",
            "Summary:",
        ]
        for item in self.corrections_summary:
            lines.append(f"  • {item}")
        
        return "\n".join(lines)


# ============================================================================
# AI MODEL ARCHITECTURE
# ============================================================================

class DiagnosticCNN(nn.Module):
    """
    Multi-head CNN for image diagnostics.
    
    Architecture:
        Input: RGB Image (B, 3, 224, 224)
        Backbone: MobileNetV2 (pretrained on ImageNet)
        Heads: 3 separate heads for exposure, white balance, saturation
    
    Outputs:
        exposure: (B, 3) - [under_prob, well_prob, over_prob]
        white_balance: (B, 5) - [color_temp_norm, tint, gain_r, gain_g, gain_b]
        saturation: (B, 3) - [under_prob, over_prob, current_level]
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Backbone: MobileNetV2 (lightweight, fast)
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Remove classifier, keep only features
        self.feature_extractor = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension from MobileNetV2
        feature_dim = 1280
        
        # Exposure Head: Classifies under/well/over exposure
        self.exposure_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # [under, well, over] logits
        )
        
        # White Balance Head: Predicts color temp + tint + gains
        self.wb_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)  # [color_temp_norm, tint, gain_r, gain_g, gain_b]
        )
        
        # Saturation Head: Predicts under/over sat + current level
        self.saturation_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # [under_prob, over_prob, current_level]
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) RGB image tensor, normalized [0, 1]
        
        Returns:
            Dict with keys: 'exposure', 'white_balance', 'saturation'
        """
        # Extract features
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Head predictions
        exposure_logits = self.exposure_head(features)
        wb_output = self.wb_head(features)
        sat_output = self.saturation_head(features)
        
        return {
            'exposure': F.softmax(exposure_logits, dim=1),  # (B, 3) probabilities
            'white_balance': wb_output,                      # (B, 5) raw values
            'saturation': torch.sigmoid(sat_output)          # (B, 3) probabilities
        }


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'diagnostic_model.pth'
_loaded_model: Optional[DiagnosticCNN] = None


def load_diagnostic_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> DiagnosticCNN:
    """
    Load the trained diagnostic model.
    
    Args:
        model_path: Path to .pth file (None = use default)
        device: Target device
    
    Returns:
        Loaded DiagnosticCNN model in eval mode
    """
    global _loaded_model
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use cached model if available
    if _loaded_model is not None:
        return _loaded_model.to(device)
    
    # Load from file
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Diagnostic model not found at {model_path}. "
            "Train the model first using train_diagnostic_model.py"
        )
    
    print(f"Loading diagnostic model from {model_path}...")
    model = DiagnosticCNN(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    _loaded_model = model
    print(f"Model loaded successfully on {device}")
    
    return model


# ============================================================================
# AI DIAGNOSTIC ANALYSIS
# ============================================================================

def _preprocess_image(img_rgb: np.ndarray, device: torch.device) -> Tensor:
    """
    Preprocess image for model input.
    
    Args:
        img_rgb: (H, W, 3) numpy array, RGB, 0-255
        device: Target device
    
    Returns:
        (1, 3, 224, 224) tensor, normalized
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H,W,C) 0-255 -> (C,H,W) 0-1
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_rgb.copy()).unsqueeze(0).to(device)
    return img_tensor


def _denormalize_color_temp(norm_value: float) -> float:
    """Convert normalized color temp [-1, 1] to Kelvin [2000, 10000]."""
    return 6000.0 + norm_value * 4000.0


def _calculate_gamma_from_exposure_probs(probs: Tensor) -> float:
    """
    Calculate gamma from exposure probabilities.
    
    Args:
        probs: (3,) tensor [under_prob, well_prob, over_prob]
    
    Returns:
        Suggested gamma value
    """
    under, well, over = probs[0].item(), probs[1].item(), probs[2].item()
    
    if under > 0.5:
        # Underexposed: increase gamma (brighten)
        gamma = 1.0 + (under - 0.5) * 1.0  # Range [1.0, 1.5]
    elif over > 0.5:
        # Overexposed: decrease gamma (darken)
        gamma = 1.0 - (over - 0.5) * 0.6  # Range [0.7, 1.0]
    else:
        # Well exposed
        gamma = 1.0
    
    return np.clip(gamma, 0.5, 2.5)


def analyze_image_with_ai(
    img_rgb: np.ndarray,
    diagnostic_model: DiagnosticCNN,
    device: torch.device
) -> Tuple[ExposureDiagnostic, WhiteBalanceDiagnostic, SaturationDiagnostic]:
    """
    Run AI diagnostic analysis on the image.
    
    Args:
        img_rgb: (H, W, 3) numpy array, RGB, 0-255
        diagnostic_model: Trained CNN model
        device: Torch device
    
    Returns:
        Tuple of (ExposureDiagnostic, WhiteBalanceDiagnostic, SaturationDiagnostic)
    """
    # Preprocess
    img_tensor = _preprocess_image(img_rgb, device)
    
    # Run model
    with torch.no_grad():
        predictions = diagnostic_model(img_tensor)
    
    # Parse exposure
    exp_probs = predictions['exposure'][0]  # (3,)
    exposure = ExposureDiagnostic(
        underexposed_probability=float(exp_probs[0]),
        overexposed_probability=float(exp_probs[2]),
        well_exposed_probability=float(exp_probs[1]),
        suggested_gamma=_calculate_gamma_from_exposure_probs(exp_probs),
        confidence=float(exp_probs.max())
    )
    
    # Parse white balance
    wb_output = predictions['white_balance'][0]  # (5,)
    color_temp_norm = float(wb_output[0])
    tint = float(wb_output[1])
    gain_r = float(torch.sigmoid(wb_output[2]) * 0.6 + 0.7)  # [0.7, 1.3]
    gain_g = float(torch.sigmoid(wb_output[3]) * 0.6 + 0.7)
    gain_b = float(torch.sigmoid(wb_output[4]) * 0.6 + 0.7)
    
    white_balance = WhiteBalanceDiagnostic(
        color_temperature_kelvin=_denormalize_color_temp(color_temp_norm),
        tint_shift=np.clip(tint, -1.0, 1.0),
        suggested_gains=(gain_r, gain_g, gain_b),
        confidence=0.85  # Can be computed from model uncertainty
    )
    
    # Parse saturation
    sat_output = predictions['saturation'][0]  # (3,)
    under_sat = float(sat_output[0])
    over_sat = float(sat_output[1])
    current_level = float(sat_output[2])
    
    # Calculate adjustment
    if under_sat > 0.5:
        sat_adjustment = 1.0 + (under_sat - 0.5) * 0.8  # [1.0, 1.4]
    elif over_sat > 0.5:
        sat_adjustment = 1.0 - (over_sat - 0.5) * 0.4  # [0.8, 1.0]
    else:
        sat_adjustment = 1.0
    
    saturation = SaturationDiagnostic(
        undersaturated_probability=under_sat,
        oversaturated_probability=over_sat,
        current_saturation_level=current_level,
        suggested_adjustment=np.clip(sat_adjustment, 0.6, 1.6),
        confidence=float(max(under_sat, over_sat))
    )
    
    return exposure, white_balance, saturation


# ============================================================================
# PARAMETER FUSION
# ============================================================================

def _synthesize_correction_params(
    scene_corrections: Dict[str, float],
    exposure: ExposureDiagnostic,
    white_balance: WhiteBalanceDiagnostic,
    saturation: SaturationDiagnostic,
    fusion_weight: float
) -> Dict[str, float]:
    """
    Fuse scene presets with AI diagnostics.
    
    Args:
        scene_corrections: Static presets from ai_scene.py
        exposure/white_balance/saturation: AI diagnostic results
        fusion_weight: Blend ratio [0, 1] (1.0 = full AI, 0.0 = full preset)
    
    Returns:
        Final parameters dict
    """
    final_params = {
        'gamma': (
            fusion_weight * exposure.suggested_gamma +
            (1 - fusion_weight) * scene_corrections['gamma']
        ),
        'gain_r': (
            fusion_weight * white_balance.suggested_gains[0] +
            (1 - fusion_weight) * scene_corrections['white_balance_r']
        ),
        'gain_g': (
            fusion_weight * white_balance.suggested_gains[1] +
            (1 - fusion_weight) * scene_corrections['white_balance_g']
        ),
        'gain_b': (
            fusion_weight * white_balance.suggested_gains[2] +
            (1 - fusion_weight) * scene_corrections['white_balance_b']
        ),
        'sat': (
            fusion_weight * saturation.suggested_adjustment +
            (1 - fusion_weight) * scene_corrections['saturation']
        ),
        'hue': scene_corrections['hue_shift']  # Keep from preset
    }
    
    # Safety clamps
    final_params['gamma'] = np.clip(final_params['gamma'], 0.5, 2.5)
    final_params['gain_r'] = np.clip(final_params['gain_r'], 0.7, 1.4)
    final_params['gain_g'] = np.clip(final_params['gain_g'], 0.7, 1.4)
    final_params['gain_b'] = np.clip(final_params['gain_b'], 0.7, 1.4)
    final_params['sat'] = np.clip(final_params['sat'], 0.6, 1.6)
    final_params['hue'] = np.clip(final_params['hue'], -0.1, 0.1)
    
    return final_params


def _generate_summary(
    exposure_pct: float,
    wb_shift_pct: float,
    sat_pct: float,
    threshold: float = 2.0
) -> List[str]:
    """Generate human-readable correction summary."""
    summary = []
    
    if abs(exposure_pct) > threshold:
        direction = "Increase" if exposure_pct > 0 else "Decrease"
        summary.append(f"{direction} exposure by {abs(exposure_pct):.1f}%")
    
    if wb_shift_pct > threshold:
        summary.append(f"Correct white balance ({wb_shift_pct:.1f}% shift detected)")
    
    if abs(sat_pct) > threshold:
        direction = "Boost" if sat_pct > 0 else "Reduce"
        summary.append(f"{direction} saturation by {abs(sat_pct):.1f}%")
    
    if not summary:
        summary.append("Image is well-balanced, minimal corrections needed")
    
    return summary


def _generate_filter_presets(final_params: Dict[str, float]) -> List[Dict[str, float]]:
    """Generate alternative filter preset variations."""
    base = final_params.copy()
    
    presets = [
        {
            'name': 'Natural',
            'params': base.copy()
        },
        {
            'name': 'Vivid',
            'params': {
                **base,
                'sat': min(base['sat'] * 1.15, 1.6),
                'gamma': base['gamma'] * 0.95
            }
        },
        {
            'name': 'Muted',
            'params': {
                **base,
                'sat': max(base['sat'] * 0.85, 0.6),
                'gamma': base['gamma'] * 1.05
            }
        }
    ]
    
    return presets


# ============================================================================
# MAIN API - Public Interface (INTERFACE CANNOT CHANGE)
# ============================================================================

def optimize_corrections(
    img_rgb: np.ndarray,
    scene_result: Dict[str, object],
    diagnostic_model: Optional[DiagnosticCNN] = None,
    device: Optional[torch.device] = None,
    fusion_weight: float = 0.6
) -> CorrectionAdvice:
    """
    Main API: Analyze image and generate correction advice.
    
    PRIMARY ENTRY POINT - Interface locked by upstream/downstream.
    
    Args:
        img_rgb: Input image (H, W, 3) numpy array, RGB, 0-255
        scene_result: Output dict from ai_scene.classify_image()
                     Must contain: 'scene', 'score', 'scene_corrections'
        diagnostic_model: Trained DiagnosticCNN (None = auto-load)
        device: Torch device
        fusion_weight: Blend ratio [0, 1] (1.0 = full AI, 0.0 = full preset)
    
    Returns:
        CorrectionAdvice object with diagnostics and parameters
    
    Example:
        >>> from src.ai_scene import classify_image
        >>> from src.ai_optimize import optimize_corrections
        >>> 
        >>> scene_result = classify_image(img_rgb)
        >>> advice = optimize_corrections(img_rgb, scene_result)
        >>> params_tensor = advice.to_tensor(device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate inputs
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be (H, W, 3), got {img_rgb.shape}")
    
    required_keys = {'scene', 'score', 'scene_corrections'}
    if not required_keys.issubset(scene_result.keys()):
        raise ValueError(f"scene_result must contain keys: {required_keys}")
    
    # Load model if not provided
    if diagnostic_model is None:
        diagnostic_model = load_diagnostic_model(device=device)
    
    # Run AI diagnostics
    exposure, white_balance, saturation = analyze_image_with_ai(
        img_rgb, diagnostic_model, device
    )
    
    # Synthesize final parameters (fusion of AI + scene presets)
    scene_corrections = scene_result['scene_corrections']
    final_params = _synthesize_correction_params(
        scene_corrections, exposure, white_balance, saturation, fusion_weight
    )
    
    # Calculate adjustment percentages
    exposure_pct = (final_params['gamma'] - 1.0) * 100
    wb_shift_pct = np.mean([
        abs(final_params['gain_r'] - 1.0),
        abs(final_params['gain_b'] - 1.0)
    ]) * 100
    sat_pct = (final_params['sat'] - 1.0) * 100
    
    # Generate human-readable summary
    summary = _generate_summary(exposure_pct, wb_shift_pct, sat_pct)
    
    # Generate filter presets
    recommended_filters = _generate_filter_presets(final_params)
    
    return CorrectionAdvice(
        scene_label=scene_result['scene'],
        scene_confidence=scene_result['score'],
        exposure=exposure,
        white_balance=white_balance,
        saturation=saturation,
        final_params=final_params,
        corrections_summary=summary,
        recommended_filters=recommended_filters,
        exposure_adjustment_percent=exposure_pct,
        white_balance_shift_percent=wb_shift_pct,
        saturation_adjustment_percent=sat_pct
    )


def generate_preview(
    img_rgb: np.ndarray,
    advice: CorrectionAdvice,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Generate preview using manual DIP pipeline.
    
    Args:
        img_rgb: Original image (H, W, 3) numpy, RGB, 0-255
        advice: CorrectionAdvice from optimize_corrections()
        device: Torch device
    
    Returns:
        Corrected preview image (H, W, 3) numpy, RGB, 0-255
    """
    if device is None:
        device = torch.device('cpu')
    
    from src.correction import apply_all_corrections_torch
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Get parameters
    params_tensor = advice.to_tensor(device)
    
    # Apply corrections
    corrected_tensor = apply_all_corrections_torch(img_tensor, params_tensor)
    
    # Convert back
    corrected_np = corrected_tensor[0].permute(1, 2, 0).cpu().numpy()
    corrected_np = (corrected_np * 255).clip(0, 255).astype(np.uint8)
    
    return corrected_np
