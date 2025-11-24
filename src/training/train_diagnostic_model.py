#!/usr/bin/env python3
# train_diagnostic_model.py

"""
Training script for DiagnosticCNN on FiveK dataset.
Trains model to predict exposure, white balance, and saturation corrections
from expert-retouched image pairs.
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.ai_optimize import DiagnosticCNN
from src.training.dataset import get_dataloaders


def compute_losses(predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute multi-head losses.
    
    Args:
        predictions: Model output dict
        batch: Ground truth batch
    
    Returns:
        Dict of losses
    """
    # Exposure loss: CrossEntropy on class probabilities
    exposure_pred = predictions['exposure']  # (B, 3) after softmax
    exposure_target = batch['exposure_label']  # (B, 3) one-hot-like probabilities
    
    # Use KL divergence for soft labels
    exposure_loss = nn.functional.kl_div(
        torch.log(exposure_pred + 1e-8),
        exposure_target,
        reduction='batchmean'
    )
    
    # White balance loss: MSE on gain predictions
    wb_pred = predictions['white_balance']  # (B, 5)
    wb_target = batch['wb_label']  # (B, 5)
    wb_loss = nn.functional.mse_loss(wb_pred, wb_target)
    
    # Saturation loss: MSE on probabilities and level
    sat_pred = predictions['saturation']  # (B, 3) after sigmoid
    sat_target = batch['sat_label']  # (B, 3)
    sat_loss = nn.functional.mse_loss(sat_pred, sat_target)
    
    # Weighted combination
    total_loss = exposure_loss + 0.8 * wb_loss + 0.6 * sat_loss
    
    return {
        'total': total_loss,
        'exposure': exposure_loss,
        'white_balance': wb_loss,
        'saturation': sat_loss
    }


def compute_metrics(predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model output dict
        batch: Ground truth batch
    
    Returns:
        Dict of metrics
    """
    # Exposure accuracy: correct class prediction
    exp_pred_class = predictions['exposure'].argmax(dim=1)
    exp_target_class = batch['exposure_label'].argmax(dim=1)
    exp_acc = (exp_pred_class == exp_target_class).float().mean().item()
    
    # White balance MAE: mean absolute error on gains
    wb_pred = predictions['white_balance'][:, 2:5]  # Extract RGB gains
    wb_target = batch['wb_label'][:, 2:5]
    wb_mae = (wb_pred - wb_target).abs().mean().item()
    
    # Saturation accuracy: within threshold
    sat_pred_adj = batch['sat_adjustment']  # We'd need to compute this from sat_pred
    # For simplicity, use MSE between sat_label predictions
    sat_mse = ((predictions['saturation'] - batch['sat_label']) ** 2).mean().item()
    
    return {
        'exposure_acc': exp_acc * 100,
        'wb_mae': wb_mae,
        'sat_mse': sat_mse
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_losses = {'total': 0.0, 'exposure': 0.0, 'white_balance': 0.0, 'saturation': 0.0}
    total_metrics = {'exposure_acc': 0.0, 'wb_mae': 0.0, 'sat_mse': 0.0}
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_images = batch['input_image'].to(device)
        for key in ['exposure_label', 'wb_label', 'sat_label', 'sat_adjustment']:
            batch[key] = batch[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(input_images)
        
        # Compute losses
        losses = compute_losses(predictions, batch)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += losses[key].item()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(predictions, batch)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={losses['total'].item():.4f}, "
                  f"exp_acc={metrics['exposure_acc']:.1f}%, "
                  f"wb_mae={metrics['wb_mae']:.4f}, "
                  f"({batches_per_sec:.1f} batch/s)")
    
    # Average metrics
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return {**avg_losses, **avg_metrics}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_losses = {'total': 0.0, 'exposure': 0.0, 'white_balance': 0.0, 'saturation': 0.0}
    total_metrics = {'exposure_acc': 0.0, 'wb_mae': 0.0, 'sat_mse': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_images = batch['input_image'].to(device)
            for key in ['exposure_label', 'wb_label', 'sat_label', 'sat_adjustment']:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            predictions = model(input_images)
            
            # Compute losses
            losses = compute_losses(predictions, batch)
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            # Compute metrics
            metrics = compute_metrics(predictions, batch)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            num_batches += 1
    
    # Average metrics
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return {**avg_losses, **avg_metrics}


def main():
    parser = argparse.ArgumentParser(description='Train DiagnosticCNN on FiveK dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to FiveK dataset directory')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--output', type=str, default='models/diagnostic_model.pth',
                        help='Output path for trained model')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Fraction of data for training (default: 0.9)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading FiveK dataset from {args.data}...")
    train_loader, val_loader = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing DiagnosticCNN...")
    model = DiagnosticCNN(pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_results = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"\nTrain: loss={train_results['total']:.4f}, "
              f"exp_acc={train_results['exposure_acc']:.1f}%, "
              f"wb_mae={train_results['wb_mae']:.4f}, "
              f"sat_mse={train_results['sat_mse']:.4f}")
        
        # Validate
        val_results = validate(model, val_loader, device)
        print(f"Val:   loss={val_results['total']:.4f}, "
              f"exp_acc={val_results['exposure_acc']:.1f}%, "
              f"wb_mae={val_results['wb_mae']:.4f}, "
              f"sat_mse={val_results['sat_mse']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_results['total'])
        
        # Save best model
        if val_results['total'] < best_val_loss:
            best_val_loss = val_results['total']
            patience_counter = 0
            print(f"✓ New best model! Saving to {output_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_results': train_results,
                'val_results': val_results,
            }, output_path)
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Final save
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
