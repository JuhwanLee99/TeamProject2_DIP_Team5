#predict.py

import torch
import numpy as np
import argparse
from src.model import ParameterPredictionModel
from src.correction import apply_all_corrections_torch
from src.io_utils import read_image, save_image
from src.visualization import plot_side_by_side
from torchvision import transforms


# --- Configuration ---
# The number of parameters your model was trained to predict
NUM_PARAMETERS = 6
# ---------------------

def preprocess(image_np, device):
    """
    Applies the same preprocessing as in the training dataset.
    image_np: (H, W, 3) NumPy array, 0-255
    """
    # (This must match your dataset's transform)
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts (H,W,C) 0-255 to (C,H,W) 0-1
        # Add resizing if your model expects a fixed size
        # transforms.Resize((256, 256)), 
        # Add normalization if used in training
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image_np.copy()).to(device)
    return tensor.unsqueeze(0) # Add batch dimension (1, C, H, W)

def postprocess(image_tensor):
    """
    Converts a tensor back to a displayable NumPy image.
    image_tensor: (C, H, W) PyTorch tensor, 0-1
    """
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
    return image_np

def predict(model_path, input_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model (Parameter Predictor)
    print(f"Loading model from {model_path}...")
    model = ParameterPredictionModel(num_params=NUM_PARAMETERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    # 2. Load and Preprocess Image
    print(f"Loading and preprocessing image from {input_path}...")
    img_bgr = read_image(input_path)
    if img_bgr is None:
        return
        
    img_rgb = img_bgr[..., ::-1] # BGR -> RGB
    input_tensor = preprocess(img_rgb, device)

    # 3. (Model) Run Inference to get PARAMETERS
    print("Running model to predict parameters...")
    with torch.no_grad():
        predicted_params = model(input_tensor) # Shape (1, NUM_PARAMETERS)

    # 4. (Correction) Apply PARAMETERS to MANUAL code
    print("Applying manual correction functions...")
    corrected_tensor = apply_all_corrections_torch(input_tensor, predicted_params)
    
    # 5. Post-process for display
    # Remove batch dimension and convert to NumPy
    corrected_image_np = postprocess(corrected_tensor[0])

    print("Displaying results...")
    plot_side_by_side(img_rgb, corrected_image_np, "Original", "Corrected (Hybrid)")
    
    # 6. Save
    output_path = 'data/output/hybrid_corrected.jpg'
    # Convert RGB NumPy image to BGR for cv2.imwrite
    save_image(output_path, corrected_image_np[..., ::-1])
    print(f"Corrected image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Hybrid ML color correction.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()
    
    predict(args.model_path, args.input_path)