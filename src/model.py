#src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterPredictionModel(nn.Module):
    def __init__(self, num_params=6):
        """
        num_params: The number of parameters we want to predict.
                    (e.g., 1_gamma + 3_wb + 1_sat + 1_hue = 6)
        """
        super(ParameterPredictionModel, self).__init__()
        
        # --- 1. Feature Extractor (e.g., a simple CNN) ---
        # This part learns to "see" the image features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 256 -> 128
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 128 -> 64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64 -> 32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        
        # --- 2. Regression Head ---
        # This part "decides" the parameter values based on the features
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_params) # Final output layer

    def forward(self, x):
        """
        x: Input image tensor (B, 3, H, W)
        """
        
        # 1. Extract features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # Shape: (B, 128, 1, 1)
        
        # Flatten the features
        x = torch.flatten(x, 1) # Shape: (B, 128)
        
        # 2. Predict parameters
        x = F.relu(self.fc1(x))
        params = self.fc2(x) # Shape: (B, num_params)
        
        # Returns raw logits. Activation (e.g., sigmoid/tanh)
        # will be applied in src/correction.py
        return params