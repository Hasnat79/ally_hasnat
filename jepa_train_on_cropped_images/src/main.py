"""
@author: Hasnat Md Abdullah
@date: Aug 28, 2025
"""
import argparse
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader


from utils import get_device_configuration
from data import LoadCroppedUnlabeledDataset
from train import JEPA_train

def main():
    parser = argparse.ArgumentParser(description= 'JEPA-based Segmentation for Cattle Depth Color-mapped + Cropped Images')

    parser.add_argument('--mode', type=str, choices=['train_jepa'], help= 'Mode: train_jepa')
    parser.add_argument('--image_dir', type=str, default='../data/unlabeled_color_mapped_cropped_imgs', help='Directory with cropped color-mapped unlabeled images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for JEPA encoder and UNet decoder training')
    parser.add_argument('--stats', action="store_true", help="Provides Dice Score and IoU for final testing, otherwise only segmentation will be performed")
    parser.add_argument('--output_dir', type=str, default = 'output_masks', help='Directory to save predicted masks')

    args = parser.parse_args()

    DEVICE = get_device_configuration()

    # Configurations settings
    WEIGHT_DIR = os.path.join(os.getcwd(), "weight")
    if not os.path.isdir(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR, exist_ok=True)
    JEPA_MODEL_NAME = 'JEPA_ENCODER_WEIGHTS.pth'
    JEPA_UNET_MODEL_NAME = 'JEPAseg_WEIGHTS.pth'

    if args.mode == 'train_jepa':
        image_dir = args.image_dir
        # check if it is directory
        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory '{image_dir}' does not exist")
        print(f"Training JEPA encoder on unlabeled color-mapped cropped images ...")
        start_time = time.time()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor()
        ])

        # loading the dataset 
        dataset = LoadCroppedUnlabeledDataset(img_dir=image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

        # Pre-train JEPA encoder on unlabeled images
        JEPA_train(dataloader,
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





if __name__ == "__main__":
    main()