# src/conversions.py

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

# ---
# Manual implementation of RGB <-> HSV conversion
# using differentiable PyTorch tensor operations.
# This satisfies the "core DIP logic" requirement for Hue/Saturation.
# Input tensors are assumed to be (B, 3, H, W) and in range 0-1.
# ---

def rgb_to_hsv_torch(rgb_tensor: Tensor) -> Tensor:
    """
    Converts an RGB tensor (B, 3, H, W) to an HSV tensor (B, 3, H, W).
    
    The output HSV channels (H, S, V) are all in the range [0, 1].
    Formulas are manually adapted for differentiable PyTorch tensor operations.
    
    Args:
        rgb_tensor: (B, 3, H, W) tensor, values in range [0, 1].
        
    Returns:
        (B, 3, H, W) tensor containing the H, S, V channels.
    """
    # Unpack channels (each is B, 1, H, W)
    r, g, b = rgb_tensor[:, 0:1, :, :], rgb_tensor[:, 1:2, :, :], rgb_tensor[:, 2:3, :, :]

    # Find max and min color component
    max_c, max_idx = torch.max(rgb_tensor, dim=1, keepdim=True) # max_c is V
    min_c = torch.min(rgb_tensor, dim=1, keepdim=True)[0]
    
    delta = max_c - min_c
    
    # --- Value (V) ---
    v = max_c
    
    # --- Saturation (S) ---
    # S = delta / V. If delta == 0 (grayscale), S = 0.
    s = torch.where(delta == 0, 
                    torch.zeros_like(delta), 
                    delta / (max_c + 1e-10)) # Add epsilon for stability
    
    # --- Hue (H) ---
    delta_plus_eps = delta + 1e-10 # Add epsilon to avoid division by zero when max_c > 0
    
    # Calculate components used for H
    # Note: These are in the range [0, 6] temporarily
    h_r = (g - b) / delta_plus_eps
    h_g = ((b - r) / delta_plus_eps) + 2.0
    h_b = ((r - g) / delta_plus_eps) + 4.0
    
    # Assign H based on which channel was the max (max_idx)
    h = torch.zeros_like(max_c)
    h = torch.where(max_idx == 0, h_r, h) # R is max -> H is (G-B)/delta
    h = torch.where(max_idx == 1, h_g, h) # G is max -> H is (B-R)/delta + 2
    h = torch.where(max_idx == 2, h_b, h) # B is max -> H is (R-G)/delta + 4
    
    # Handle grayscale (delta == 0): Hue is undefined, set to 0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    
    # Normalize H from [0, 6] to [0, 1] via cyclic division
    h = (h / 6.0) % 1.0
    
    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb_torch(hsv_tensor: Tensor) -> Tensor:
    """
    Converts an HSV tensor (B, 3, H, W) to an RGB tensor (B, 3, H, W).
    
    Input H, S, V channels are expected to be in the range [0, 1].
    Formulas are manually adapted for differentiable PyTorch tensor operations.
    
    Args:
        hsv_tensor: (B, 3, H, W) tensor containing the H, S, V channels, range [0, 1].
        
    Returns:
        (B, 3, H, W) tensor containing the R, G, B channels, range [0, 1].
    """
    # Unpack channels (each is B, 1, H, W)
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
    
    # Scale H from [0, 1] to [0, 6] for sector calculation
    h_scaled = h * 6.0
    h_i = h_scaled.floor() # Integer part (hue sector, 0-5)
    f = h_scaled - h_i      # Fractional part
    
    # Calculate intermediate values (p, q, t)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    
    # Hue sector as integer (0-5)
    h_i_int = h_i.long() % 6
    
    # Create an empty RGB tensor (B, 3, H, W)
    rgb = torch.zeros_like(hsv_tensor)
    
    # Assign R, G, B components based on the hue sector (h_i_int)
    # Sector 0: (V, T, P) -> R=V, G=T, B=P
    rgb = torch.where(h_i_int == 0, torch.cat([v, t, p], dim=1), rgb)
    # Sector 1: (Q, V, P) -> R=Q, G=V, B=P
    rgb = torch.where(h_i_int == 1, torch.cat([q, v, p], dim=1), rgb)
    # Sector 2: (P, V, T) -> R=P, G=V, B=T
    rgb = torch.where(h_i_int == 2, torch.cat([p, v, t], dim=1), rgb)
    # Sector 3: (P, Q, V) -> R=P, G=Q, B=V
    rgb = torch.where(h_i_int == 3, torch.cat([p, q, v], dim=1), rgb)
    # Sector 4: (T, P, V) -> R=T, G=P, B=V
    rgb = torch.where(h_i_int == 4, torch.cat([t, p, v], dim=1), rgb)
    # Sector 5: (V, P, Q) -> R=V, G=P, B=Q
    rgb = torch.where(h_i_int == 5, torch.cat([v, p, q], dim=1), rgb)
    
    # Handle grayscale (s == 0): R=G=B=V
    rgb = torch.where(s == 0, torch.cat([v, v, v], dim=1), rgb)
    
    return rgb