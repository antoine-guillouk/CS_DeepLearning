import os
import numpy as np
from PIL import Image
from PIL import ImageDraw 
import torch

def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

class ImgDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 masked_dir,
                 unmasked_dir,
                 augmentation = None,
                 preprocessing = None):

        self.masked_image_paths = [os.path.join(masked_dir, masked_image_id) for masked_image_id in sorted(os.listdir(masked_dir)) if masked_image_id.endswith('.png')]
        self.unmasked_image_paths = [os.path.join(unmasked_dir, unmask_image_id) for unmask_image_id in sorted(os.listdir(unmasked_dir)) if unmask_image_id.endswith('.png')]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # read images and masks
        unmasked_image = pil_loader(self.unmasked_image_paths[index])
        masked_image = pil_loader(self.masked_image_paths[index])
        
        # apply augmentations
        if self.augmentation:
            unmasked_image = self.augmentation(unmasked_image)
            masked_image = self.augmentation(masked_image)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=unmasked_image, mask=masked_image)
            unmasked_image, masked_image = sample['image'], sample['mask']
            
        return unmasked_image, masked_image

    def __len__(self):
        return len(self.unmasked_image_paths)
