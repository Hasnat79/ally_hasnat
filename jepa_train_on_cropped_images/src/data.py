"""
@author: Hasnat Md Abdullah
@date: Aug 28, 2025
"""
from torch.utils.data import Dataset
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils import normalize_pixel_depth_values_to_rgb_array

class LoadCroppedUnlabeledDataset(Dataset):
    
    def __init__(self, img_dir, transform):
        super().__init__()
        """
        Args:
            img_dir (str): Directory of cropped & color mapped cattle unlabeled (no mask) images (.png)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

class LoadLabeledDataset(Dataset):

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

        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
        self.img_files.sort()

        if len(self.img_files) == 0:
            raise ValueError(f"No .tif or .tiff files found in {img_dir}")
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image_name = self.img_files[idx]
        base_name = os.path.splitext(image_name)[0]  # e.g. "103L_5424"
        img_path = os.path.join(self.img_dir, image_name)
        mask_path = os.path.join(self.mask_dir,f"{base_name}_mask.png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found for image {image_name} at {mask_path}")
        
        # --- Load and preprocess image ---
        image_raw = Image.open(img_path)
        original_size = image_raw.size # (width, height)
        image = normalize_pixel_depth_values_to_rgb_array(image_raw)
        if image is None:
            raise ValueError(f"Failed to normalize image {image_name}")
        image = np.array(image) # RGB_array


        # load mask 
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.uint8)

        # apply paired argumentations 
        if self.transform:
            augmented = self.transform(image=image,mask = mask)
            image = augmented['image'] #tensor [C,H,W]
            mask = augmented['mask'] #tensor [H,W] or [1,H,W] depending on ToTensorV2

        else:
            image = ToTensorV2()(image=image)['image']
            mask = ToTensorV2()(image=mask)['image']

        # Ensure mask has shape [1,H,W] instead of [H,W]
        if mask.ndim == 2:  # H,W
            mask = mask.unsqueeze(0)   # -> 1,H,W
            
        if self.stage == 'train':
            return image, mask
        else:
            return image, mask, image_name, original_size






if __name__ == "__main__":
    dataset = LoadCroppedUnlabeledDataset(img_dir="../data/unlabeled_color_mapped_cropped_imgs", transform=None)
    print(f"Dataset size: {len(dataset)}")