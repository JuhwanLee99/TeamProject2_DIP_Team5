#!/usr/bin/env python3
# train_diagnostic_model.py

"""
Training script for DiagnosticCNN on FiveK dataset.
Trains model to predict exposure, white balance, and saturation corrections
from expert-retouched image pairs.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
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
    SIMPLIFIED loss computation - complex losses were causing instability.
    
    Args:
        predictions: Model output dict
        batch: Ground truth batch
    
    Returns:
        Dict of losses
    """
    # Exposure loss: Simple CrossEntropy (stable and proven)
    exposure_pred = predictions['exposure']  # (B, 3) softmax probabilities
    exposure_target = batch['exposure_label']  # (B, 3) soft labels
    
    # Use CrossEntropy with soft targets
    exposure_loss = -(exposure_target * torch.log(exposure_pred + 1e-8)).sum(dim=1).mean()
    
    # White balance loss: Simple MSE (stable)
    wb_pred = predictions['white_balance'][:, 2:5]  # (B, 3) RGB gains only
    wb_target = batch['wb_label'][:, 2:5]  # (B, 3) RGB gains
    wb_loss = nn.functional.mse_loss(wb_pred, wb_target)
    
    # Saturation loss: Simple MSE on adjustment value only
    sat_pred = predictions['saturation'][:, 2]  # (B,) single value
    sat_target = batch['sat_adjustment'].squeeze()  # (B,) target adjustment
    sat_loss = nn.functional.mse_loss(sat_pred, sat_target)
    
    # Balanced combination (equal weights - simpler is better)
    total_loss = exposure_loss + wb_loss + sat_loss
    
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
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None,
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch with mixed precision and gradient accumulation."""
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
        
        # Mixed precision training
        # Simple training loop
        optimizer.zero_grad()
        predictions = model(input_images)
        losses = compute_losses(predictions, batch)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32, effective 128 with grad accum)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001 - higher for faster convergence)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--output', type=str, default='models/new/diagnostic_model.pth',
                        help='Output path for trained model')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Fraction of data for training (default: 0.9)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'mobilenet'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps (default: 4, effective batch=128)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Use mixed precision training (default: False - disabled for stability)')
    parser.add_argument('--swa', action='store_true', default=False,
                        help='Use Stochastic Weight Averaging (default: False - disabled for simplicity)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 0.0001)')
    parser.add_argument('--disable-early-stop', action='store_true', default=False,
                        help='Disable early stopping to train full epochs')
    
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
    
    # Create model with selected backbone
    print(f"\nInitializing DiagnosticCNN with {args.backbone} backbone...")
    model = DiagnosticCNN(pretrained=True, backbone=args.backbone).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple SGD optimizer (more stable than AdamW for this task)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("✓ Using mixed precision training (FP16)")
    
    # Simple ReduceLROnPlateau scheduler (proven to work)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    print("✓ Using ReduceLROnPlateau scheduler")
    
    # SWA setup (optional)
    swa_model = None
    swa_start_epoch = args.epochs // 2 if args.swa else args.epochs + 1
    if args.swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        print(f"✓ Using SWA (starts at epoch {swa_start_epoch})")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10  # Reasonable patience
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=None, gradient_accumulation_steps=1  # Disabled for stability
        )
        print(f"\nTrain: loss={train_results['total']:.4f}, "
              f"exp_acc={train_results['exposure_acc']:.1f}%, "
              f"wb_mae={train_results['wb_mae']:.4f}, "
              f"sat_mse={train_results['sat_mse']:.4f}")
        
        # Validate
        val_model = swa_model if args.swa and epoch >= swa_start_epoch else model
        val_results = validate(val_model, val_loader, device)
        print(f"Val:   loss={val_results['total']:.4f}, "
              f"exp_acc={val_results['exposure_acc']:.1f}%, "
              f"wb_mae={val_results['wb_mae']:.4f}, "
              f"sat_mse={val_results['sat_mse']:.4f}")
        
        # Learning rate scheduling - simple ReduceLROnPlateau
        if args.swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
        scheduler.step(val_results['exposure_acc'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR = {current_lr:.6f}")
        
        # Save best model (prioritize accuracy over loss)
        is_best = False
        if val_results['exposure_acc'] > best_val_acc:
            best_val_acc = val_results['exposure_acc']
            is_best = True
        if val_results['total'] < best_val_loss:
            best_val_loss = val_results['total']
            is_best = True
        
        if is_best:
            patience_counter = 0
            print(f"✓ New best model! (acc={best_val_acc:.1f}%, loss={best_val_loss:.4f})")
            print(f"  Saving to {output_path}")
            
            # Save the SWA model if active, otherwise regular model
            save_model = swa_model if args.swa and epoch >= swa_start_epoch else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'train_results': train_results,
                'val_results': val_results,
                'backbone_type': args.backbone,
                'config': vars(args)
            }, output_path)
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
        
        # Early stopping (but not before SWA completes, or disabled)
        if not args.disable_early_stop:
            if patience_counter >= patience and epoch >= swa_start_epoch + 15:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        else:
            if patience_counter >= patience:
                print(f"Would early stop, but --disable-early-stop is set. Continuing...")
                patience_counter = 0  # Reset to continue
    
    # Final SWA batch normalization update
    if args.swa:
        print("\nUpdating SWA batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Final validation with updated SWA model
        print("Final SWA validation...")
        final_val_results = validate(swa_model, val_loader, device)
        print(f"Final SWA Val: loss={final_val_results['total']:.4f}, "
              f"exp_acc={final_val_results['exposure_acc']:.1f}%, "
              f"wb_mae={final_val_results['wb_mae']:.4f}")
        
        # Save final SWA model
        if final_val_results['exposure_acc'] > best_val_acc or final_val_results['total'] < best_val_loss:
            print(f"✓ SWA improved results! Saving final model...")
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': swa_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': final_val_results['total'],
                'val_acc': final_val_results['exposure_acc'],
                'val_results': final_val_results,
                'backbone_type': args.backbone,
                'config': vars(args),
                'swa_applied': True
            }, output_path)
            best_val_acc = final_val_results['exposure_acc']
            best_val_loss = final_val_results['total']
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Total epochs: {epoch + 1}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
