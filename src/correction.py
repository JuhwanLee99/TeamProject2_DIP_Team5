# src/correction.py

import torch
import torch.nn.functional as F
from torch import Tensor

# ---
# Import conversion functions
# ---
from src.conversions import rgb_to_hsv_torch, hsv_to_rgb_torch

# ---
# CORE DIP LOGIC
# Implemented using manual PyTorch operations to maintain a 
# fully differentiable computation graph for training.
# ---

# --- RGB Space Functions ---

def apply_gamma_torch(img_tensor: Tensor, gamma: Tensor) -> Tensor:
    """
    Applies gamma correction to the input tensor.
    
    Args:
        img_tensor: (B, C, H, W) tensor, values in range [0, 1].
        gamma: (B, 1) tensor of gamma values.
    """
    gamma_expanded = gamma.view(-1, 1, 1, 1)
    # Avoid division by zero or instability
    gamma_expanded = torch.clamp(gamma_expanded, min=1e-6)
    
    corrected = torch.pow(img_tensor, 1.0 / gamma_expanded)
    return torch.clamp(corrected, 0.0, 1.0)

def apply_white_balance_torch(img_tensor: Tensor, gains: Tensor) -> Tensor:
    """
    Applies channel-wise gains for white balance.
    
    Args:
        img_tensor: (B, 3, H, W) tensor, values in range [0, 1].
        gains: (B, 3) tensor of (R, G, B) gains.
    """
    gains_expanded = gains.view(-1, 3, 1, 1)
    balanced = img_tensor * gains_expanded
    return torch.clamp(balanced, 0.0, 1.0)

# --- HSV Space Functions ---

def apply_saturation_torch(hsv_tensor: Tensor, sat_adjustment: Tensor) -> Tensor:
    """
    Adjusts saturation channel in HSV space.
    
    Args:
        hsv_tensor: (B, 3, H, W) tensor [H, S, V].
        sat_adjustment: (B, 1) tensor (scaling factor).
    """
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
    sat_adj_expanded = sat_adjustment.view(-1, 1, 1, 1)
    
    s_new = torch.clamp(s * sat_adj_expanded, 0.0, 1.0)
    return torch.cat([h, s_new, v], dim=1)

def apply_hue_torch(hsv_tensor: Tensor, hue_rotation: Tensor) -> Tensor:
    """
    Adjusts hue via cyclic rotation.
    
    Args:
        hsv_tensor: (B, 3, H, W) tensor [H, S, V].
        hue_rotation: (B, 1) tensor (shift in range [0, 1]).
    """
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
    hue_rot_expanded = hue_rotation.view(-1, 1, 1, 1)
    
    # Cyclic shift within [0, 1]
    h_new = (h + hue_rot_expanded) % 1.0 
    return torch.cat([h_new, s, v], dim=1)

# --- Main Correction Pipeline ---

def apply_all_corrections_torch(img_tensor: Tensor, params_tensor: Tensor) -> Tensor:
    """
    Applies the full correction pipeline based on model predictions.
    
    Args:
        img_tensor: (B, 3, H, W) input image tensor (0-1).
        params_tensor: (B, 6) tensor containing:
            [gamma, gain_r, gain_g, gain_b, sat, hue]
    """
    # Unpack parameters
    gamma = params_tensor[:, 0:1]         # (B, 1)
    gains = params_tensor[:, 1:4]         # (B, 3)
    sat = params_tensor[:, 4:5]           # (B, 1)
    hue = params_tensor[:, 5:6]           # (B, 1)

    # 1. White Balance (Linear space)
    img = apply_white_balance_torch(img_tensor, gains)
    
    # 2. Gamma Correction
    img = apply_gamma_torch(img, gamma)
    
    # 3. Convert RGB -> HSV
    hsv = rgb_to_hsv_torch(img)
    
    # 4. Adjust Saturation
    hsv = apply_saturation_torch(hsv, sat)
    
    # 5. Adjust Hue
    hsv = apply_hue_torch(hsv, hue)
    
    # 6. Convert HSV -> RGB
    img_corr = hsv_to_rgb_torch(hsv)
    
    # Final safety clamp
    return torch.clamp(img_corr, 0.0, 1.0)