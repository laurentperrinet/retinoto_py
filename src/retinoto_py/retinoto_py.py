"""Main module."""

#############################################################
#############################################################
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms


# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 


def make_mask(image_size:int, radius:float = 0.5):
    """Create a circular mask for the image.
    image_size: int, size of the image (height and width)
    radius: float, radius of the circle (0.5 means half the image size)"""
    X, Y = np.meshgrid(np.linspace(-radius, radius, image_size, endpoint=True), 
               np.linspace(-radius, radius, image_size, endpoint=True))
    R = np.sqrt(X**2 + Y**2)
    mask = (R < 0.5).astype(np.float32)
    return torch.from_numpy(mask)

class ApplyMask: 
    """Apply a mask to the image."""
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, images):
        return images[:, :, ::] * self.mask

# class CleanRotations_class(object): 
#     """Apply a rotation to the image.
#     Clean because apply a circular mask to the image so the border does not reveal the rotation."""
#     def __init__(self, angles):
#         self.angles = angles

#     def __call__(self, images):
#         temp = []
#         images = images.unsqueeze(dim=0) if len(images) == 3 else images
#         for image in images:
#             for angle in self.angles:
#                 temp.append(T.functional.rotate(image, angle=angle, expand = False)) 
#         return torch.stack(temp)
    

def get_preprocess(args):
    # --- 5. Define Image Pre-processing ---
    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    preprocess = transforms.Compose([
        transforms.Resize(args.image_size),                # Resize the shortest side to 256px
        transforms.CenterCrop(args.image_size),            # Crop the center 224x224 pixels
        transforms.ToTensor(),                 # Convert the image to a PyTorch Tensor
        transforms.Normalize(                  # Normalize with ImageNet mean and std
            mean=im_mean,
            std=im_std
        )
    ])

    return preprocess

