import torchvision.transforms as transforms

# Spatial size of training images. All images will be resized to this
#   size using a transformer.

def get__augmentation():
    transform = [transforms.Scale(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
    return transforms.Compose(transform)

