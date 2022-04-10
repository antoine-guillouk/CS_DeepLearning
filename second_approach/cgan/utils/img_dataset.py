import os
import numpy as np
from PIL import Image
from PIL import ImageDraw 
import torch
import cv2

def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

class ImgDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                ground_truth_dir,
                masked_img_dir,
                mask_map_dir,
                augmentation = None,
                preprocessing = None):

        self.ground_truth_paths = [os.path.join(ground_truth_dir, ground_truth_id) for ground_truth_id in sorted(os.listdir(ground_truth_dir)) if ground_truth_id.endswith('.png')]
        self.masked_image_paths = [os.path.join(masked_img_dir, masked_image_id) for masked_image_id in sorted(os.listdir(masked_img_dir)) if masked_image_id.endswith('.png')]
        self.mask_map_paths = [os.path.join(mask_map_dir, mask_map_id) for mask_map_id in sorted(os.listdir(mask_map_dir)) if mask_map_id.endswith('.png')]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # read images and masks
        ground_truth_image = pil_loader(self.ground_truth_paths[index])
        masked_image = cv2.cvtColor(cv2.imread(self.masked_image_paths[index]), cv2.COLOR_BGR2RGB)
        mask_map = cv2.imread(self.mask_map_paths[index])
        inpainting_img = Image.fromarray(cv2.subtract(masked_image, mask_map))
        
        
        # apply augmentations
        if self.augmentation:
            ground_truth_image = self.augmentation(ground_truth_image)
            inpainting_img = self.augmentation(inpainting_img)
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=ground_truth_image, inpainting_img=inpainting_img)
            ground_truth_image, inpainting_img = sample['image'], sample['inpainting_image']
            
        return ground_truth_image, inpainting_img

    def __len__(self):
        return len(self.masked_image_paths)
