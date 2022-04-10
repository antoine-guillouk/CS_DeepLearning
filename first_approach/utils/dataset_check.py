import os
import sys
import pathlib

working_directory = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(working_directory))

from utils.load_config import load_config


config = load_config()
dataroot = config["dataroot"]
unmasked_img_dir = os.path.join(dataroot, 'unmasked')
masked_img_dir = os.path.join(dataroot, 'masked')

def dataset_check():
    mask, unmask = set(), set()
    u, n = 0, 0

    for file in os.listdir(unmasked_img_dir):
        unmask.add(file.replace('.png',''))
        u+=1

    for file in os.listdir(masked_img_dir):
        mask.add((file[:5]))
        n+=1

    print("Datasets checked")

    assert len(unmask - mask) == 0, f"{unmask - mask} are not in both dataset"
