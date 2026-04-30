# Normalization, resizing to 256x256, transforming in tensor
from torchvision import transforms

def get_transform(image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), #interpolation: BILINEAR
        transforms.ToTensor() #returns a tensor like [channel, H, W] with values [0.,1.]
        ])

    return transform