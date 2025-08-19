"""
Image Anomaly Detection

@author: Bresolin Lab
Email: tgbresolin@gmail.com
"""
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from src.data import LoadUnlabelDataset
from src.data import LoadLabeledData
from src.train import JEPAtrain
from src.train import SEGtrain
from src.inference import testing
from src.inference import inference
from src.utils import summarize_results

parser = argparse.ArgumentParser(description='JEPA-based Segmentation for Cattle Depth Images')
parser.add_argument('--mode', type=str, choices=['train_jepa', 'train_unet', 'inference'],
                    help='Mode: train_jepa, train_unet, or inference')
parser.add_argument('--image_dir', type=str, default='path/to/labeled/depth',
                    help='Directory with labeled depth images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, 
                    help='Epochs for JEPA encoder and UNet decoder training')
parser.add_argument('--stats', action="store_true", help="Provides Dice Score and IoU \
                    for final testing, otherwise only segmentation will be performed")
parser.add_argument('--output_dir', type=str, default='output_masks',
                    help='Directory to save predicted masks')
args = parser.parse_args()

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f'Using cuda and device: "{torch.cuda.get_device_name(0)}"')
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using mps")
else:
    DEVICE = torch.device("cpu")

# Configurations settings
WEIGHT_DIR = os.path.join(os.getcwd(), "weight")
if not os.path.isdir(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR, exist_ok=True)
JEPA_MODEL_NAME = 'JEPA_ENCODER_WEIGHTS.pth'
JEPA_UNET_MODEL_NAME = 'JEPAseg_WEIGHTS.pth'

if args.mode == 'train_jepa':
        
    # Set image directory
    image_dir = args.image_dir
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory '{image_dir}' not found.")
        
    print("Training JEPA encoder on unlabeled images ...")
    start_time = time.time()
    
    # Set the image transformation for unleabeled images while loading
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(30),
    #     transforms.ToTensor(),  
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor()
    ])
    
    # Load data
    dataset = LoadUnlabelDataset(image_dir, transform) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
    # Pre-train JEPA encoder on unlabeled images
    JEPAtrain(dataloader, 
              DEVICE,
              args.epochs,
              WEIGHT_DIR,
              JEPA_MODEL_NAME,
              proj_dim=256,
              lr=1e-4, 
              imgformat='rgb',
              momentum=0.99,
              pretrained=True)
    
    # Calculate elapsed time to calculate PCs
    elapsed_time = time.time() - start_time
    print(f"Computational to train JEPA: {elapsed_time:.2f}s") 

if args.mode == 'train_unet':
    
    # Set image directory
    image_dir = args.image_dir
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory '{image_dir}' not found.")
    
    # mask_dir = args.mask_dir
    # if not os.path.isdir(mask_dir):
    #     raise FileNotFoundError(f"Directory '{mask_dir}' not found.")
        
    # Encoder weights path
    jepa_weights_path = os.path.join(WEIGHT_DIR, JEPA_MODEL_NAME)
        
    print("Training UNet-like dencoder on labeled images ...")
    start_time = time.time()
    
    # Set the image transformation for training leabeled images while loading
    train_transform = A.Compose([
        A.Resize(224, 224),             # Resize image and mask
        A.HorizontalFlip(p=0.5),        # Random horizontal flip
        A.VerticalFlip(p=0.5),          # Random vertical flip
        A.Rotate(limit=30, p=0.5),      # Random rotation
        A.RandomBrightnessContrast(p=0.3),  # optional color jitter
        ToTensorV2()                    # Convert both to tensors
        ])
    
    # Set the image transformation for valiation Leabeled images while loading
    valid_transform = A.Compose([
        A.Resize(224, 224),  # Resize to 224x224
        ToTensorV2()  # Convert to PyTorch tensor
    ])
    
    train_image_dir = os.path.join(image_dir, 'train/image')
    train_mask_dir = os.path.join(image_dir, 'train/mask')
    valid_image_dir = os.path.join(image_dir, 'valid/image')
    valid_mask_dir = os.path.join(image_dir, 'valid/mask')
    
    # Create datasets with the custom mapping
    train_dataset = LoadLabeledData(train_image_dir, train_mask_dir, 'train', transform=train_transform)
    valid_dataset = LoadLabeledData(valid_image_dir, valid_mask_dir, 'train', transform=valid_transform)
    
    # Create DataLoaders    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Train UNet decoder like on labeled data
    SEGtrain(train_loader,
             valid_loader,
             DEVICE,
             jepa_weights_path,
             WEIGHT_DIR,
             JEPA_UNET_MODEL_NAME,
             proj_dim=256,
             out_classes=1,
             imgformat='rgb',
             lr=1e-3,
             epochs=args.epochs,
             patience=8)
    
    # Calculate elapsed time for training UNet decoder
    elapsed_time = time.time() - start_time
    print(f"\nComputational time for fine tunning: {elapsed_time:.2f}s") 
    

if args.mode == 'inference':
    
    print("Inference JEPAUNet ...")
    start_time = time.time()
    
    # Encoder+Decoder weights path
    jepa_unet_weights_path = os.path.join(WEIGHT_DIR, JEPA_UNET_MODEL_NAME)
    
    # Image transformation
    transform = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

    if args.stats:
        
        # Set directories
        image_dir = os.path.join(args.image_dir, 'test/image')
        mask_dir = os.path.join(args.image_dir, 'test/mask')
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Directory '{image_dir}' not found.")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Directory '{mask_dir}' not found.")

        
        # Create directory to save segmentation
        seg_dir = os.path.join(args.image_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)
        
        # Create datasets with the custom mapping and load it
        dataset = LoadLabeledData(image_dir, mask_dir, 'test', transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        results = testing(jepa_unet_weights_path, dataloader, DEVICE, seg_dir)
        pd.DataFrame(results).to_csv(os.path.join(os.getcwd(), 'results.csv'), index=False)
        print(f"Results saved to {os.path.join(os.getcwd(), 'results.csv')}")

        summarize_results(results)
 
#     else:
#         # Inference-only (no ground-truth masks)
#         dataset = LoadInferenceDataset(IMAGE_ROOT_DIR, transform, args.imgformat)
#         dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#         inference(model, dataloader, DEVICE, seg_dir)
