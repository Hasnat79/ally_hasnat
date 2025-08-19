#
# Created on Tue Aug 19 2025 at 1:04:49 PM
#
# @author: Hasnat Md Abdullah
#

from data_loader import UnlabeledColorMappedDataset, save_image
from config import UNLABELED_DATAPATH, COLOR_MAP_IMG_OUTPUT_DIR
from tqdm import tqdm

if __name__ == "__main__":

    unlabeled_color_mapped_dataset = UnlabeledColorMappedDataset(UNLABELED_DATAPATH)

    # Process each image in the dataset
    for i in tqdm(range(len(unlabeled_color_mapped_dataset)), desc="Processing unlabeled images"):
        # Get the color mapped image
        image = unlabeled_color_mapped_dataset[i]
        # Save the color mapped image
        save_image(COLOR_MAP_IMG_OUTPUT_DIR, image, unlabeled_color_mapped_dataset.img_files[i])