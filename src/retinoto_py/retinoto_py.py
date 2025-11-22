"""Main module."""

#############################################################
#############################################################
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 


def make_mask(image_size: int, radius: float = 1.0):
    """
    Create a circular mask for the image.
    
    image_size: int, size of the image (height and width)
    radius: float, radius of the circle (0.5 means half the image size)"""
    
    X, Y = np.meshgrid(np.linspace(-1, 1, image_size), # Coordonnées normalisées de -1 à 1
                       np.linspace(-1, 1, image_size),
                       indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    mask = (R <= radius).astype(np.float32) # 1.0 pour un cercle complet
    return torch.from_numpy(mask).unsqueeze(0) # Ajoute la dimension du canal

class ApplyMask(transforms.PILToTensor):
    """Applique un masque circulaire à un tenseur d'image."""
    def __init__(self, mask: torch.Tensor):
        # On stocke le masque. Le .clone() est une bonne pratique pour éviter
        # des modifications inattendues du masque original.
        self.mask = mask.clone()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applique le masque à un tenseur d'image.
        Args:
            tensor (torch.Tensor): Tenseur d'image de forme (C, H, W).
        Returns:
            torch.Tensor: Tenseur masqué.
        """
        return tensor * self.mask
    

def get_preprocess(args, angle_min=None, angle_max=None):
    # --- 5. Define Image Pre-processing ---
    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    transform_list = [
                        transforms.Resize(args.image_size),                # Resize the shortest side to 256px
                        transforms.CenterCrop(args.image_size),            # Crop the center 224x224 pixels
                        ]

    # Si les deux angles ne sont pas None, on applique la rotation
    if angle_min is not None and angle_max is not None:
        transform_list.append(transforms.RandomRotation(degrees=(angle_min, angle_max)))
    
    transform_list.append(transforms.ToTensor())  # Convert the image to a PyTorch Tensor

    if args.do_mask:
        # Créer le masque une seule fois avec la taille de l'image
        mask = make_mask(image_size=args.image_size)
        # Ajouter notre transform personnalisée à la liste
        transform_list.append(ApplyMask(mask))

    # Ajouter la normalisation (toujours à la fin)
    transform_list.append(transforms.Normalize(mean=im_mean, std=im_std))

    # Créer la chaîne de prétraitement finale
    preprocess = transforms.Compose(transform_list)
    return preprocess
    
from tqdm.auto import tqdm
def get_validation_accuracy(args, model, val_loader):
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for images, true_labels in tqdm(val_loader, desc=f"Evaluating {args.model_name}"):
        images = images.to(args.device)
        true_labels = true_labels.to(args.device)

        # Get predictions (no need for gradients)
        with torch.no_grad():
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, dim=1)

        # Check if the prediction was correct for the entire batch
        # The comparison produces a tensor of booleans (True/False)
        correct_predictions_in_batch = (predicted_labels == true_labels)

        # Sum the boolean tensor to get the number of correct predictions in the batch
        # .item() extracts the number from the tensor
        correct_predictions += correct_predictions_in_batch.sum().item()

        # The total number of predictions is the batch size
        total_predictions += true_labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy