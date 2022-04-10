import torch
import cv2
import os

from encoding import one_hot_encode
from utils import get_data_dir

class MaskDataset(torch.utils.data.Dataset):

    """Trees Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            fold='train', 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,

    ):
        dir = get_data_dir(fold)
        self.image_paths = [os.path.join(dir["x"], image_id) for image_id in sorted(os.listdir(dir["x"])) if image_id.endswith('.png')]
        self.mask_paths = [os.path.join(dir["y"], mask_id) for mask_id in sorted(os.listdir(dir["y"])) if mask_id.endswith('.png')]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.COLOR_GRAY2RGB)

        mask = cv2.merge((mask,mask,mask))
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)