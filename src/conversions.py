#src/conversions.py

import torch
import torch.nn.functional as F

# ---
# Manual implementation of RGB <-> HSV conversion
# using differentiable PyTorch tensor operations.
# This satisfies the "core DIP logic" requirement for Hue/Saturation.
# Input tensors are assumed to be (B, 3, H, W) and in range 0-1.
# ---

def rgb_to_hsv_torch(rgb_tensor):
    """
    Converts RGB tensor (B, 3, H, W) to HSV tensor (B, 3, H, W).
    Formulas adapted for PyTorch tensors.
    """
    r, g, b = rgb_tensor[:, 0:1, :, :], rgb_tensor[:, 1:2, :, :], rgb_tensor[:, 2:3, :, :]

    max_c, max_idx = torch.max(rgb_tensor, dim=1, keepdim=True)
    min_c = torch.min(rgb_tensor, dim=1, keepdim=True)[0]
    
    delta = max_c - min_c
    
    # --- Value (V) ---
    v = max_c
    
    # --- Saturation (S) ---
    # if delta == 0, s = 0 (grayscale)
    s = torch.where(delta == 0, 
                    torch.zeros_like(delta), 
                    delta / (max_c + 1e-10)) # Add epsilon for stability
    
    # --- Hue (H) ---
    delta_plus_eps = delta + 1e-10 # Add epsilon to avoid division by zero
    
    # Calculate R, G, B components for hue
    h_r = (g - b) / delta_plus_eps
    h_g = ((b - r) / delta_plus_eps) + 2.0
    h_b = ((r - g) / delta_plus_eps) + 4.0
    
    # Assign based on which channel was max
    h = torch.zeros_like(max_c)
    h = torch.where(max_idx == 0, h_r, h) # R is max
    h = torch.where(max_idx == 1, h_g, h) # G is max
    h = torch.where(max_idx == 2, h_b, h) # B is max
    
    # Handle grayscale (delta == 0)
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    
    # Normalize H to 0-1 (from 0-6)
    h = (h / 6.0) % 1.0
    
    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb_torch(hsv_tensor):
    """
    Converts HSV tensor (B, 3, H, W) to RGB tensor (B, 3, H, W).
    Formulas adapted for PyTorch tensors.
    """
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]
    
    # Scale H from 0-1 to 0-6
    h_i = (h * 6.0).floor()
    f = (h * 6.0) - h_i
    
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    
    h_i_int = h_i.long() % 6
    
    # Create an empty RGB tensor
    rgb = torch.zeros_like(hsv_tensor)
    
    # Assign R, G, B based on hue sector (h_i_int)
    rgb = torch.where(h_i_int == 0, torch.cat([v, t, p], dim=1), rgb)
    rgb = torch.where(h_i_int == 1, torch.cat([q, v, p], dim=1), rgb)
    rgb = torch.where(h_i_int == 2, torch.cat([p, v, t], dim=1), rgb)
    rgb = torch.where(h_i_int == 3, torch.cat([p, q, v], dim=1), rgb)
    rgb = torch.where(h_i_int == 4, torch.cat([t, p, v], dim=1), rgb)
    rgb = torch.where(h_i_int == 5, torch.cat([v, p, q], dim=1), rgb)
    
    # Handle grayscale (s == 0)
    rgb = torch.where(s == 0, torch.cat([v, v, v], dim=1), rgb)
    
    return rgb