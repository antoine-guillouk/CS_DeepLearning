import os
import numpy as np
from PIL import Image
from PIL import ImageDraw 


root = './data'
rgb = os.path.join(root, 'RGB')
gt = os.path.join(root, 'GT')
dirs = [root, rgb, gt]

for dir_ in dirs:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

# Generate some images
n = 8
for i in range(n):
    arr = np.zeros((32, 32, 3))

    arr[:,:,:] = np.random.randint(0, 255, 3)
    im1 = Image.fromarray(np.uint8(arr))
    # Add some text
    ImageDraw.Draw(im1).text((0, 0), 'RGB',(255, 255, 255))
    im1.save(os.path.join(rgb, f'img{i+1}.png'))

    im2 = Image.fromarray(np.uint8(arr))
    ImageDraw.Draw(im2).text((0, 0), 'GT',(255, 255, 255))
    im2.save(os.path.join(gt, f'img{i+1}.png'))

def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []

    # Our dir names
    rgb_dir = 'RGB'
    gt_dir = 'GT'   
    
    # Get all the filenames from RGB folder
    rgb_fnames = sorted(os.listdir(os.path.join(root, rgb_dir)))
    
    # Compare file names from GT folder to file names from RGB:
    for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):

            if gt_fname in rgb_fnames:
                # if we have a match - create pair of full path to the corresponding images
                rgb_path = os.path.join(root, rgb_dir, gt_fname)
                gt_path = os.path.join(root, gt_dir, gt_fname)

                item = (rgb_path, gt_path)
                # append to the list dataset
                dataset.append(item)
            else:
                continue

    return dataset

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset


class CustomVisionDataset(VisionDataset):
    
    def __init__(self,
                 root,
                 loader=default_loader,
                 rgb_transform=None,
                 gt_transform=None):
        super().__init__(root,
                         transform=rgb_transform,
                         target_transform=gt_transform)

        # Prepare dataset
        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples
        # list of RGB images
        self.rgb_samples = [s[1] for s in samples]
        # list of GT images
        self.gt_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        rgb_path, gt_path = self.samples[index]
        
        # import each image using loader (by default it's PIL)
        rgb_sample = self.loader(rgb_path)
        gt_sample = self.loader(gt_path)
        
        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        if self.transform is not None:
            rgb_sample = self.transform(rgb_sample)
        if self.target_transform is not None:
            gt_sample = self.target_transform(gt_sample)      
        
        # now we return the right imported pair of images (tensors)
        return rgb_sample, gt_sample

    def __len__(self):
        return len(self.samples)

from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


bs=4  # batch size
transforms = ToTensor()  # we need this to convert PIL images to Tensor
shuffle = True

dataset = CustomVisionDataset('./data', rgb_transform=transforms, gt_transform=transforms)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)

for i, (rgb, gt) in enumerate(dataloader):
    print(f'batch {i+1}:')
    # some plots
    for i in range(bs):
        plt.figure(figsize=(10, 5))
        plt.subplot(221)
        plt.imshow(rgb[i].squeeze().permute(1, 2, 0))
        plt.title(f'RGB img{i+1}')
        plt.subplot(222)
        plt.imshow(gt[i].squeeze().permute(1, 2, 0))
        plt.title(f'GT img{i+1}')
        plt.show()