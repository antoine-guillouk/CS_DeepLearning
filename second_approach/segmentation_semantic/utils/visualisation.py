## Helper functions for viz. & one-hot encoding/decoding

import matplotlib.pyplot as plt
import random

from encoding import colour_code_segmentation, reverse_one_hot
from mask_dataset import MaskDataset
from utils import get_class_dict

# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def visualize_dataset(id=None, augmentation=None, fold="train"):
    class_rgb_values = get_class_dict()['color']

    dataset = MaskDataset(
        fold, 
        augmentation=augmentation,
        class_rgb_values=class_rgb_values,
    )

    if id is None:
        id = random.randint(0, len(dataset)-1)

    #Different augmentations on a random image/mask pair (256*256 crop)
    nb_test = 1 if augmentation is None else 3
    for i in range(nb_test):
        image, mask = dataset[id]
        visualize(
            original_image = image,
            ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), class_rgb_values),
            one_hot_encoded_mask = reverse_one_hot(mask)
        )
