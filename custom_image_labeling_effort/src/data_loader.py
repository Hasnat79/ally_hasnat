#
# Created on Tue Aug 19 2025 at 12:08:55 PM
#
# @author: Hasnat Md Abdullah
#
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

def get_color_mapped_image(image, floor_distance=3000):
    """
        #TODO: why do we need floor distance?

    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    # print("Image shape:", image.shape) #(480, 848)
    
    #
    # boolean mask for pixels: True if the pixel value is greater than or equal to floor_distance
    # This will be used to filter out invalid pixels
    # and to create a color mapped image
    # masking out certain regions: potentially the floor
    #
    mask = image >= floor_distance
    image[mask] = 0  # Set pixels >= floor_distance to 0
    valid_pixels_having_cattle = image > 0  # Identify valid pixels

    # skip the image
    if not np.any(valid_pixels_having_cattle):
        print("No valid pixels found in the image. Skipping...")
        return None

    depth_min, depth_max = np.min(image[valid_pixels_having_cattle]), np.max(image[valid_pixels_having_cattle])

    # Adjust depth_max slightly to avoid division by zero
    if depth_min == depth_max:
        depth_max += 1e-5

    depth_norm = np.zeros(image.shape, dtype=np.uint8)
    depth_norm[valid_pixels_having_cattle] = ((image[valid_pixels_having_cattle] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET) # convert to color img (RGB)
    depth_norm = cv2.cvtColor(depth_norm, cv2.COLOR_BGR2RGB)
    #
    # print("Color mapped image shape:", depth_norm.shape) #(480, 848, 3)
    #
    return Image.fromarray(depth_norm)

def save_image(output_dir, image, tiff_image_path): 
    """ takes tiff file name, saves as png"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Extract the base name without extension
    base_name = os.path.splitext(os.path.basename(tiff_image_path))[0]
    # Construct the output file path
    output_file_path = os.path.join(output_dir, f"{base_name}.png")
    # Save the image
    image.save(output_file_path)
class UnlabeledColorMappedDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)    

    def __getitem__(self,idx):
        image_path = os.path.join(self.img_dir, self.img_files[idx])

        image = Image.open(image_path)

        #TODO: convert the raw image to color mapped image
        image = get_color_mapped_image(image)

        #TODO: transform not implemented yet
        if self.transform:
            image = self.transform(image)
        return image
if __name__ == "__main__":
    # Example usage
    unlabeled_dataset = UnlabeledDataset(UNLABELED_DATAPATH)
    print(f"Number of unlabeled images: {len(unlabeled_dataset)}") #3387

    # traverse unlabeled images
    for i in tqdm(range(len(unlabeled_dataset)), desc="Processing images"):
        image = unlabeled_dataset[i]
        save_image(COLOR_MAP_IMG_OUTPUT_DIR, image, unlabeled_dataset.img_files[i])

    print("Done processing unlabeled images.")
