import os
import PIL
from PIL import Image

unmasked_img_full_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/unmasked'
masked_img_full_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/masked'

unmasked_img_red_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/red/unmasked'
masked_img_red_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/red/masked'

def reduce_dataset_unmasked():
    unmask = set()
    for file in os.listdir(unmasked_img_full_dir):
        if int(file[0]) < 4:
            unmask.add(file)
        
    for file in unmask:
        print(unmasked_img_full_dir + r'/' + file)
        picture = Image.open(unmasked_img_full_dir + r'/' + file)
        picture = picture.save(unmasked_img_red_dir + r'/' + file)


def reduce_dataset_masked():
    mask = set()
    for file in os.listdir(masked_img_full_dir):
        if int(file[0]) < 4:
            mask.add(file)

    for file in mask:
        print(masked_img_full_dir + r'/' + file)
        picture = Image.open(masked_img_full_dir + r'/' + file)
        picture = picture.save(masked_img_red_dir + r'/' + file)

reduce_dataset_masked()