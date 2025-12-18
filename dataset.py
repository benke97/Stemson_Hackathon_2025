import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import torchvision.transforms.functional as TF

class HAADFDataset(Dataset):
    def __init__(self, h5_path, split='train', train_ratio=0.9, crops_per_image=10):
        """
        Args:
            h5_path (str): Path to h5 dataset
            split (str): 'train' or 'val'
            train_ratio (float): Percentage of data to use for training
            crops_per_image (int): Number of crops generated per original image (default 10)
        """
        self.h5_path = h5_path
        self.split = split
        
        with h5py.File(h5_path, 'r') as f:
            if 'images' not in f:
                raise KeyError(f"Dataset 'images' not found in {h5_path}")
            total_crops = f['images'].shape[0]
            
        # --- PREVENT DATA LEAKAGE ---
        # 1. Calculate number of unique original images
        num_unique_images = total_crops // crops_per_image
        
        if total_crops % crops_per_image != 0:
            print(f"Warning: Total samples ({total_crops}) is not divisible by crops_per_image ({crops_per_image}).")

        # 2. Create indices for UNIQUE IMAGES (not crops)
        all_image_indices = np.arange(num_unique_images)
        
        # 3. Shuffle the IMAGES
        np.random.seed(67) 
        np.random.shuffle(all_image_indices)
        
        # 4. Split based on images
        split_idx = int(num_unique_images * train_ratio)
        
        if split == 'train':
            selected_image_indices = all_image_indices[:split_idx]
        else:
            selected_image_indices = all_image_indices[split_idx:]
            
        # 5. Expand back to CROP indices
        # Example: If Image 5 is selected, we add crops 50, 51, ... 59
        self.indices = []
        for img_idx in selected_image_indices:
            start_crop = img_idx * crops_per_image
            end_crop = start_crop + crops_per_image
            self.indices.extend(range(start_crop, end_crop))
            
        self.indices = np.array(self.indices)
        
        print(f"Dataset loaded: {split} split with {len(self.indices)} samples "
              f"({len(selected_image_indices)} unique source images).")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.dset_images = self.h5_file['images']
            self.dset_metadata = self.h5_file['metadata']

        real_idx = self.indices[idx]

        img_np = self.dset_images[real_idx]     # Shape: (224, 224)
        meta_np = self.dset_metadata[real_idx]  # Shape: (14,)

        # Duplicate 3 times: (3, 224, 224) to mimic RGB (For ResNet18)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).repeat(3, 1, 1)
        meta_tensor = torch.from_numpy(meta_np)

        # Augment only for training
        if self.split == 'train':
            img_tensor = self.augment(img_tensor)

        return img_tensor, meta_tensor

    def augment(self, img):
        """
        Applies random flips and 90-degree rotations.
        """
        if random.random() > 0.5:
            img = TF.hflip(img)

        k = random.randint(0, 3) 
        if k > 0:
            img = torch.rot90(img, k, [1, 2])
            
        return img