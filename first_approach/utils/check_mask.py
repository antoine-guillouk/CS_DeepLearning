import os

unmasked_img_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/unmasked'
masked_img_dir = r'/gpfs/workdir/baillyv/deepL/data/cGAN/masked'

def remove_unmasked_img():
    unmask = set()
    mask = set()
    for file in os.listdir(unmasked_img_dir):
        unmask.add(file)
        
    for file in os.listdir(masked_img_dir):
        mask.add(file.replace('_surgical',''))
        
    to_remove = unmask - mask
    for file in to_remove:
        os.remove(os.path.join(unmasked_img_dir,file))

remove_unmasked_img()