"""
@author: Hasnat Md Abdullah
@date: Aug 28, 2025
"""
from torch.utils.data import Dataset
import os
from PIL import Image

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

if __name__ == "__main__":
    dataset = LoadCroppedUnlabeledDataset(img_dir="../data/unlabeled_color_mapped_cropped_imgs", transform=None)
    print(f"Dataset size: {len(dataset)}")