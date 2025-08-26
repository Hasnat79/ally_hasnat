#
# Created on Tue Aug 19 2025 at 1:04:49 PM
#
# @author: Hasnat Md Abdullah
#
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import UnlabeledColorMappedDataset, save_image
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color mapped images from unlabeled TIFF data.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=f"Path to the input unlabeled TIFF data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Path to the output directory for color mapped images "
    )
    args = parser.parse_args()

    unlabeled_color_mapped_dataset = UnlabeledColorMappedDataset(args.input_dir)

    # Process each image in the dataset
    for i in tqdm(range(len(unlabeled_color_mapped_dataset)), desc="Processing unlabeled images"):
        # Get the color mapped image
        image = unlabeled_color_mapped_dataset[i]
        # Save the color mapped image
        save_image(args.output_dir, image, unlabeled_color_mapped_dataset.img_files[i])