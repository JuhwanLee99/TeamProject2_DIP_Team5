#train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ColorDataset
from src.model import ParameterPredictionModel
from src.correction import apply_all_corrections_torch

# --- Configuration ---
DATA_DIR = 'data/INTEL-TAU' # Path to your dataset
MODEL_SAVE_PATH = 'models/hybrid_model.pth'
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# The number of parameters your model will predict
# Example: 1(gamma) + 3(R,G,B gains) + 1(saturation) + 1(hue) = 6
NUM_PARAMETERS = 6 
# ---------------------

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset and DataLoader
    print("Initializing dataset...")
    # (Note: You must implement the logic in src/dataset.py)
    train_dataset = ColorDataset(root_dir=DATA_DIR, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # (Optional) Validation loader
    # val_dataset = ColorDataset(root_dir=DATA_DIR, split='val')
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataset loaded. Found {len(train_dataset)} training images.")

    # 2. Model
    model = ParameterPredictionModel(num_params=NUM_PARAMETERS).to(device)

    # 3. Loss Function and Optimizer
    # We compare the *final images*, so MSE or L1 loss is appropriate
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting hybrid training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for input_images, target_images in progress_bar:
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            
            # --- This is the Hybrid Logic ---
            
            # 1. (Model) Predict parameters
            #    Output shape: (BATCH_SIZE, NUM_PARAMETERS)
            predicted_params = model(input_images)
            
            # 2. (Correction) Apply parameters using manual PyTorch code
            #    This step is fully differentiable
            corrected_images = apply_all_corrections_torch(input_images, predicted_params)
            
            # 3. (Loss) Compare corrected image with the ground-truth target
            loss = criterion(corrected_images, target_images)
            
            # 4. (Optimize) Backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ----------------------------------
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
        
        # (TODO: Add a validation loop here)

    print("Training finished.")
    
    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()