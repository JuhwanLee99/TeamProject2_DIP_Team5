# src/training/__init__.py

"""
Training module for DiagnosticCNN model.
"""

from .dataset import FiveKDataset, get_dataloaders
from .train_diagnostic_model import main as train_main

__all__ = ['FiveKDataset', 'get_dataloaders', 'train_main']
