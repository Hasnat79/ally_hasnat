"""
Image Anomaly Detection

@author: Bresolin Lab
Email: tgbresolin@gmail.com
"""
import os
import numpy as np
#import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
from src.utils import processing


class LoadUnlabelDataset(Dataset):
    """
    Custom Dataset for JEPA encoder training supporting both grayscale and RGB images.
    """
    def __init__(self, img_dir, transform):
        super().__init__()
        """
        Args:
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path)
        
        try:
            image = processing(image)
        except ValueError as e:
            raise ValueError(f"Processing failed for {img_path}: {str(e)}")

        if self.transform:
            image = self.transform(image)
            
        return image

class LoadLabeledData(Dataset):
    """
    Custom Dataset for segmentation training.
    Dataset for segmentation: loads depth TIFFs, preprocesses with `processing()`,
    and matches them with mask files that have '_mask.png' suffix.
    """
    def __init__(self, img_dir, mask_dir, stage, transform=None):
        """
        Args:
            img_dir (str): Directory containing input .tiff images.
            mask_dir (str): Directory containing masks with '_mask.png' suffix.
            transform (albumentations.Compose): Transform pipeline (applied to image and mask).
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.stage = stage
        self.transform = transform

        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))]
        self.img_files.sort()

        if len(self.img_files) == 0:
            raise ValueError(f"No TIFF images found in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]  # e.g. "103L_5424"
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {img_name}. Expected: {mask_path}")

        # --- Load and preprocess image ---
        image_raw = Image.open(img_path)
        original_size = image_raw.size  # (width, height)
        image = processing(image_raw)
        if image is None:
            raise ValueError(f"Processing failed or no valid pixels in image: {img_path}")
        image = np.array(image)  # RGB array
        
         # --- Load mask ---
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.uint8)
                
        # --- Apply paired augmentations ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # tensor [C,H,W]
            mask = augmented['mask']    # tensor [H,W] or [1,H,W] depending on ToTensorV2
        else:
            image = ToTensorV2()(image=image)['image']
            mask = ToTensorV2()(image=mask)['image']
        
        # Ensure mask has shape [1,H,W] instead of [H,W]
        if mask.ndim == 2:  # H,W
            mask = mask.unsqueeze(0)   # -> 1,H,W
            
        if self.stage == 'train':
            return image, mask
        else:
            return image, mask, img_name, original_size
