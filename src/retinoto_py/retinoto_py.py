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

import time
import pandas as pd
def train_model(args, model, train_loader, val_loader, df_train=None, each_steps:int=64, 
                verbose:bool=True, do_save:bool=True, model_filename='resnet.pth'):
    
    # retraining the full model
    for param in model.parameters():
        param.requires_grad = True        


    # sets the optimizer
    if args.delta2 > 0.: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(1-args.delta1, 1-args.delta2), weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=1-args.delta1, weight_decay=args.weight_decay) # to set training variables
    
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    criterion = torch.nn.CrossEntropyLoss()

    # the DataFrame to record from
    if df_train is None:
        i_epoch_start = 0
        df_train = pd.DataFrame([], columns=['epoch', 'i_image', 'total_image', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'time']) 
        if verbose: print(f"Starting learning...")
    else:
        i_epoch_start = df_train['epoch'].max() + 1
        if verbose: print(f"Starting from epoch {i_epoch_start} with {len(df_train)} records")
        # # reset the index
        # df_train.reset_index(drop=True, inplace=True)
        # make a copy of the DataFrame to avoid modifying the original one
        df_train = df_train.copy()

    since = time.time()
    total_image = 0
    n_train = len(train_loader.dataset)
    n_train_stop = args.n_train_stop
    if n_train_stop==0: n_train_stop = n_train

    avg_loss_ = avg_acc_ = []
    for i_epoch in range(i_epoch_start, args.num_epochs):
        i_image = 0
        for i_step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            total_image += len(images)
            i_image += len(images)
            if i_image > n_train_stop: break # early stopping

            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            optimizer.zero_grad(set_to_none=True)
            # for param in model.parameters():
            #     param.grad = None

            outputs = model(images)
             
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, dim=1)

            avg_loss_.append(loss.item() * images.size(0))
            avg_acc_.append(torch.mean((preds == labels.data)*1.).cpu().item()) # append average accuracy in the last batch

            if (i_step % (max(n_train_stop//args.batch_size//each_steps, 1))==0) : #or (i_step == n_train_stop-1):
                with torch.no_grad():

                    avg_loss = np.mean(avg_loss_)
                    avg_acc = np.mean(avg_acc_)

                    loss_val = 0
                    acc_val = 0
                    model = model.eval()
                    n_val = len(val_loader)
                    for _, (images, labels) in enumerate(val_loader):
                        images, labels = images.to(args.device), labels.to(args.device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)

                        loss_val += loss.item() * images.size(0)

                        _, preds = torch.max(outputs.data, dim=1)
                        acc_val += torch.mean((preds == labels.data)*1.).cpu().item()

                    avg_loss_val = loss_val / n_val
                    avg_acc_val = acc_val / n_val

                    df_train.loc[len(df_train)] = {'epoch': i_epoch, 'i_image':i_image, 'total_image':total_image, 'avg_loss':avg_loss, 'avg_acc':avg_acc, 'avg_loss_val':avg_loss_val, 'avg_acc_val':avg_acc_val, 'time':time.time() - since}
                    if verbose:  print(f"{model_filename} - Epoch {i_epoch}, i_image {i_image} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f} / time:{time.time() - since:.1f}")
                avg_loss_ = avg_acc_ = []

        if do_save:
            if verbose:  print(f"Saving...{model_filename}")
            torch.save(model.state_dict(), model_filename)
            df_train.to_json(model_filename.replace('pth', 'json'), orient='index', indent=2)


    return model, df_train