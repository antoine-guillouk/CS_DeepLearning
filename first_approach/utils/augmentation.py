import torchvision.transforms as transforms

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 512

def get__augmentation():
    transform = [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
    return transforms.Compose(transform)

