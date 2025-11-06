#src/dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.io_utils import read_image
import glob

class ColorDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir (string): Directory with the dataset (e.g., 'data/INTEL-TAU').
            split (string): 'train' or 'val'.
        """
        
        # --- (Needs Implementation) ---
        # This class MUST be implemented to find pairs of
        # (input_image, ground_truth_image)
        #
        # Example for a dataset structured like:
        # root_dir/
        #   train/
        #     input/
        #       0001.png
        #     ground_truth/
        #       0001.png
        #
        # self.input_files = sorted(glob.glob(os.path.join(root_dir, split, 'input', '*')))
        # self.target_files = sorted(glob.glob(os.path.join(root_dir, split, 'ground_truth', '*')))
        
        # Placeholder:
        self.input_files = [] 
        self.target_files = []
        if not self.input_files:
            print(f"Warning: Dataset at {root_dir} is not implemented or empty. Found 0 files.")
            print("Please implement the file path logic in src/dataset.py")

        # Define transformations
        # (Must be the same for input and target, unless adding data augmentation)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts (H,W,C) 0-255 to (C,H,W) 0-1
            # (Optional) Resize if images are different sizes
            # transforms.Resize((256, 256)), 
            # (Optional) Normalize if needed
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        # return len(self.input_files)
        return 10 # Placeholder length (Remove this when implemented)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # --- (Needs Implementation) ---
        # 1. Load input image
        # input_path = self.input_files[idx]
        # input_img = read_image(input_path)[..., ::-1] # BGR -> RGB
        
        # 2. Load target image
        # target_path = self.target_files[idx]
        # target_img = read_image(target_path)[..., ::-1] # BGR -> RGB
        
        # 3. Apply transforms
        # input_tensor = self.transform(input_img.copy())
        # target_tensor = self.transform(target_img.copy())
        
        # Placeholder data (Remove this when implemented):
        input_tensor = torch.rand(3, 256, 256)
        target_tensor = torch.rand(3, 256, 256)
        
        return input_tensor, target_tensor