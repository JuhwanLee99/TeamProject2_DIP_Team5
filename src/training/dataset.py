# src/dataset.py

"""
FiveK Dataset loader for training diagnostic model.
Loads input/expert pairs and computes ground truth correction parameters.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def compute_gamma_correction(input_img: np.ndarray, target_img: np.ndarray) -> float:
    """
    Compute gamma correction needed to transform input to target.
    
    Args:
        input_img: (H, W, 3) RGB array [0, 1]
        target_img: (H, W, 3) RGB array [0, 1]
    
    Returns:
        Gamma value that best transforms input to target
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    input_safe = np.clip(input_img, eps, 1.0)
    target_safe = np.clip(target_img, eps, 1.0)
    
    # Compute gamma: target = input^gamma
    # gamma = log(target) / log(input)
    log_input = np.log(input_safe)
    log_target = np.log(target_safe)
    
    # Compute per-pixel gamma and take median (robust to outliers)
    gamma_map = log_target / (log_input + eps)
    gamma = float(np.median(gamma_map))
    
    return np.clip(gamma, 0.5, 2.5)


def compute_white_balance_gains(input_img: np.ndarray, target_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute RGB gain corrections needed to transform input to target.
    
    Args:
        input_img: (H, W, 3) RGB array [0, 1]
        target_img: (H, W, 3) RGB array [0, 1]
    
    Returns:
        (gain_r, gain_g, gain_b) multipliers
    """
    eps = 1e-8
    
    # Compute mean per channel
    input_mean = input_img.mean(axis=(0, 1)) + eps
    target_mean = target_img.mean(axis=(0, 1)) + eps
    
    # Gain = target_mean / input_mean
    gains = target_mean / input_mean
    
    # Normalize so green channel = 1.0 (standard white balance reference)
    gains = gains / gains[1]
    
    # Clip to reasonable range
    gains = np.clip(gains, 0.7, 1.4)
    
    return float(gains[0]), float(gains[1]), float(gains[2])


def compute_saturation_adjustment(input_img: np.ndarray, target_img: np.ndarray) -> float:
    """
    Compute saturation adjustment needed to transform input to target.
    
    Args:
        input_img: (H, W, 3) RGB array [0, 1]
        target_img: (H, W, 3) RGB array [0, 1]
    
    Returns:
        Saturation multiplier
    """
    # Compute saturation as std dev in RGB space
    input_sat = np.std(input_img, axis=2).mean()
    target_sat = np.std(target_img, axis=2).mean()
    
    eps = 1e-8
    sat_adjustment = (target_sat + eps) / (input_sat + eps)
    
    return np.clip(float(sat_adjustment), 0.6, 1.6)


def compute_exposure_class(gamma: float) -> Tuple[float, float, float]:
    """
    Convert gamma correction to exposure classification with SIMPLE thresholds.
    
    Args:
        gamma: Computed gamma correction value
    
    Returns:
        (under_prob, well_prob, over_prob)
    """
    # Simple, clear thresholds
    if gamma > 1.3:
        # Clearly underexposed
        return 0.9, 0.05, 0.05
    elif gamma > 1.1:
        # Slightly underexposed
        return 0.7, 0.25, 0.05
    elif gamma < 0.7:
        # Clearly overexposed
        return 0.05, 0.05, 0.9
    elif gamma < 0.9:
        # Slightly overexposed
        return 0.05, 0.25, 0.7
    else:
        # Well exposed
        return 0.1, 0.8, 0.1


def compute_saturation_class(sat_adj: float) -> Tuple[float, float]:
    """
    SIMPLE saturation classification - not used in new model (uses direct regression).
    
    Args:
        sat_adj: Computed saturation adjustment value
    
    Returns:
        (under_prob, over_prob)
    """
    # Simple binary classification
    if sat_adj > 1.15:
        return 0.8, 0.1
    elif sat_adj < 0.85:
        return 0.1, 0.8
    else:
        return 0.1, 0.1


class FiveKDataset(Dataset):
    """
    FiveK dataset loader for training diagnostic model.
    
    Loads input/expert pairs and computes ground truth correction parameters.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.9,
        image_size: int = 224,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Path to FiveK dataset root
            split: 'train' or 'val'
            train_ratio: Fraction of data for training
            image_size: Target image size for model input
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Find input and expert directories
        self.input_dir = self._find_input_dir()
        self.expert_dir = self._find_expert_dir()
        
        # Load image pairs
        self.image_pairs = self._load_image_pairs()
        
        # Split train/val
        num_train = int(len(self.image_pairs) * train_ratio)
        if split == 'train':
            self.image_pairs = self.image_pairs[:num_train]
        else:
            self.image_pairs = self.image_pairs[num_train:]
        
        # Transforms
        self.transform = self._build_transforms()
    
    def _find_input_dir(self) -> Path:
        """Find input images directory."""
        candidates = [
            self.data_dir / 'raw',
            self.data_dir / 'input',
            self.data_dir / 'source',
            self.data_dir / 'original',
        ]
        for path in candidates:
            if path.exists():
                return path
        raise ValueError(f"Could not find input directory in {self.data_dir}")
    
    def _find_expert_dir(self) -> Path:
        """Find expert retouched images directory."""
        candidates = [
            self.data_dir / 'c',
            self.data_dir / 'expertC',
            self.data_dir / 'expert_c',
            self.data_dir / 'target',
            self.data_dir / 'expertC_output',
        ]
        for path in candidates:
            if path.exists():
                return path
        raise ValueError(f"Could not find expert directory in {self.data_dir}")
    
    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load and match input/expert image pairs."""
        input_images = sorted(list(self.input_dir.glob('*.png')) + list(self.input_dir.glob('*.jpg')))
        
        pairs = []
        for input_path in input_images:
            # Try to find matching expert file
            expert_path = self.expert_dir / input_path.name
            if not expert_path.exists():
                # Try with different extension
                expert_path = self.expert_dir / (input_path.stem + '.png')
            if not expert_path.exists():
                expert_path = self.expert_dir / (input_path.stem + '.jpg')
            
            if expert_path.exists():
                pairs.append((input_path, expert_path))
        
        if len(pairs) == 0:
            raise ValueError(f"No matching input/expert pairs found in {self.data_dir}")
        
        return pairs
    
    def _build_transforms(self) -> transforms.Compose:
        """Build image transformations with enhanced augmentation."""
        transform_list = []
        
        if self.augment:
            # Enhanced training augmentations for better generalization
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),  # Vertical flip sometimes
                transforms.RandomRotation(15),  # Small rotation for robustness
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                # Stronger color augmentation to learn exposure/WB variations
                transforms.ColorJitter(
                    brightness=0.3,  # Increased for exposure variation
                    contrast=0.3,    # Increased for dynamic range
                    saturation=0.2,  # Increased for saturation learning
                    hue=0.05         # Small hue shift
                ),
                transforms.RandomGrayscale(p=0.05),  # Occasionally grayscale
                # Add random blur for robustness
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                if np.random.random() < 0.3 else transforms.Lambda(lambda x: x),
            ])
        else:
            # Validation: just center crop to ensure consistent size
            transform_list.append(transforms.CenterCrop(self.image_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load image pair and compute ground truth labels.
        
        Returns:
            Dict with keys:
                - input_image: (3, 224, 224) tensor
                - exposure_label: (3,) [under, well, over]
                - exposure_gamma: (1,) target gamma
                - wb_label: (5,) [temp_norm, tint, gain_r, gain_g, gain_b]
                - sat_label: (3,) [under_prob, over_prob, current_level]
                - sat_adjustment: (1,) target saturation adjustment
        """
        input_path, expert_path = self.image_pairs[idx]
        
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        expert_img = Image.open(expert_path).convert('RGB')
        
        # OPTIMIZATION: Resize to target size BEFORE computing ground truth
        # This is 10-20x faster than computing on full resolution images
        target_size = (self.image_size, self.image_size)
        input_img_small = input_img.resize(target_size, Image.LANCZOS)
        expert_img_small = expert_img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy for ground truth computation (on small images)
        input_np = np.array(input_img_small).astype(np.float32) / 255.0
        expert_np = np.array(expert_img_small).astype(np.float32) / 255.0
        
        # Compute ground truth correction parameters (fast on 224x224)
        gamma = compute_gamma_correction(input_np, expert_np)
        gain_r, gain_g, gain_b = compute_white_balance_gains(input_np, expert_np)
        sat_adj = compute_saturation_adjustment(input_np, expert_np)
        
        # Convert to classification labels
        exp_under, exp_well, exp_over = compute_exposure_class(gamma)
        sat_under, sat_over = compute_saturation_class(sat_adj)
        
        # Compute current saturation level (0 to 1)
        current_sat = float(np.std(input_np, axis=2).mean())
        
        # Apply transforms (already resized, so this just does augmentations if training)
        input_tensor = self.transform(input_img_small)
        
        # Prepare labels
        exposure_label = torch.tensor([exp_under, exp_well, exp_over], dtype=torch.float32)
        exposure_gamma = torch.tensor([gamma], dtype=torch.float32)
        
        # Normalize color temp to [-1, 1] range
        # Assume gains encode temp: warmer = higher R, cooler = higher B
        temp_indicator = (gain_r - gain_b) / 0.5  # Normalized difference
        tint = 0.0  # FiveK doesn't directly encode tint
        
        wb_label = torch.tensor([temp_indicator, tint, gain_r, gain_g, gain_b], dtype=torch.float32)
        
        sat_label = torch.tensor([sat_under, sat_over, current_sat], dtype=torch.float32)
        sat_adjustment = torch.tensor([sat_adj], dtype=torch.float32)
        
        return {
            'input_image': input_tensor,
            'exposure_label': exposure_label,
            'exposure_gamma': exposure_gamma,
            'wb_label': wb_label,
            'sat_label': sat_label,
            'sat_adjustment': sat_adjustment,
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    train_ratio: float = 0.9
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to FiveK dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Fraction of data for training
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = FiveKDataset(data_dir, split='train', train_ratio=train_ratio, augment=True)
    val_dataset = FiveKDataset(data_dir, split='val', train_ratio=train_ratio, augment=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
