"""Main module."""

#############################################################
#############################################################
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm

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
    transform_list = []
    transform_list.append(transforms.Resize(args.image_size))
    transform_list.append(transforms.CenterCrop(args.image_size))
                        
    # Si les deux angles ne sont pas None, on applique la rotation
    if angle_min is not None and angle_max is not None:
        transform_list.append(transforms.RandomRotation(degrees=(angle_min, angle_max)))
    
    transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.append(transforms.ToTensor())  # Convert the image to a PyTorch Tensor
    transform_list.append(transforms.Normalize(mean=im_mean, std=im_std))

    if args.do_mask:
        # Créer le masque une seule fois avec la taille de l'image
        mask = make_mask(image_size=args.image_size)
        # Ajouter notre transform personnalisée à la liste
        transform_list.append(ApplyMask(mask))

    # Créer la chaîne de prétraitement finale
    preprocess = transforms.Compose(transform_list)
    return preprocess
    
def get_validation_accuracy(args, model, val_loader, desc=None):
    if desc is None:
        desc = f"Evaluating {args.model_name}"

    model = model.to(args.device)
    n_val_stop = args.n_val_stop
    if n_val_stop==0: n_val_stop = len(val_loader.dataset)

    model.eval()

    correct_predictions = 0
    total_predictions = 0

    outer_progress = tqdm(val_loader, desc=desc, total=n_val_stop//args.batch_size)

    for images, true_labels in outer_progress:
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
        if total_predictions > n_val_stop: break # early stopping


    accuracy = correct_predictions / total_predictions
    outer_progress.set_postfix(f"accuracy={accuracy:.4f}")

    return accuracy

import time
import pandas as pd
from tqdm.auto import tqdm

def train_model(args, model, train_loader, val_loader, df_train=None, #each_steps=64, 
                verbose=True, do_save=True, 
                model_filename='resnet.pth', json_filename='resnet.json'):
    
    model = model.to(args.device)
    # retraining the full model
    for param in model.parameters():
        param.requires_grad = True        

    n_train_stop = args.n_train_stop
    if n_train_stop==0: n_train_stop = len(train_loader.dataset)
    n_val_stop = args.n_val_stop
    if n_val_stop==0: n_val_stop = len(val_loader.dataset)

    # sets the optimizer
    if args.delta2 > 0.: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(1-args.delta1, 1-args.delta2), 
                                      weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=1-args.delta1, 
                                    weight_decay=args.weight_decay) # to set training variables
    
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    criterion = torch.nn.CrossEntropyLoss()

    # the DataFrame to record from
    if df_train is None:
        i_epoch_start = 0
        if verbose: print(f"Starting learning...")
    else:
        i_epoch_start = df_train['epoch'].max() + 1
        if verbose: print(f"Starting from epoch {i_epoch_start} with {len(df_train)} records")
        # # reset the index
        # df_train.reset_index(drop=True, inplace=True)
        # make a copy of the DataFrame to avoid modifying the original one
        df_train = df_train.copy()

    since = time.time()
    history = []
    total_image = 0
    outer_progress = tqdm(range(i_epoch_start, args.num_epochs), desc="Epochs", leave=False)
    for i_epoch in outer_progress:
        running_loss = 0.0
        running_corrects = 0
        i_image = 0
        for i_batch, (images, true_labels) in tqdm(enumerate(train_loader), desc=f'epoch={i_epoch+1}/{args.num_epochs}', total=n_train_stop//args.batch_size, leave=True):

            model.train()

            images, true_labels = images.to(args.device), true_labels.to(args.device)
            total_image += len(images)
            i_image += len(images)
            if i_image > n_train_stop: break # early stopping

            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            optimizer.zero_grad(set_to_none=True)
            # for param in model.parameters():
            #     param.grad = None

            outputs = model(images)
            _, predicted_labels = torch.max(outputs, dim=1)
            running_corrects += (predicted_labels == true_labels).sum().item()
             
            loss = criterion(outputs, true_labels)
            running_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()


        loss_train = running_loss / i_image
        acc_train = running_corrects*1. / i_image

        model.eval()  # Set model to evaluation mode
        running_loss_val = 0.0
        running_corrects_val = 0
        i_image = 0
        with torch.no_grad():
            for images, true_labels in val_loader:
                images, true_labels = images.to(args.device), true_labels.to(args.device)
                outputs = model(images)
                _, predicted_labels = torch.max(outputs, dim=1)
                running_corrects_val += (predicted_labels == true_labels).sum().item()

                loss = criterion(outputs, true_labels)
                running_loss_val += loss.item() * images.size(0)
                i_image += len(images)
                if i_image > n_val_stop: break # early stopping


        loss_val = running_loss_val / n_val_stop
        acc_val = running_corrects_val / n_val_stop

        outer_progress.set_postfix(f"Acc: train={acc_train:.4f} - val={acc_val:.4f}")
        history.append({'epoch': i_epoch, 'i_image':i_image, 'total_image':total_image, 'loss_train':loss_train, 'acc_train':acc_train, 'loss_val':loss_val, 'acc_val':acc_val, 'time':time.time() - since})
        # if verbose:  print(f"{model_filename} \t| Epoch {i_epoch}, i_image {i_image} \t| train= loss: {loss_train:.4f} \t| acc : {acc_train:.4f} - val= loss : {loss_val:.4f} \t| acc : {acc_val:.4f} \t| time:{time.time() - since:.1f}")

    if df_train is None:
        df_train = pd.DataFrame(history)
    else:
        df_new_row = pd.DataFrame(history)
        df_train = pd.concat([df_train, df_new_row], ignore_index=True)
    if do_save:
        if verbose:  print(f"Saving...{model_filename}")
        torch.save(model.state_dict(), model_filename)
        df_train.to_json(json_filename, orient='records', indent=2)


    return model, df_train

def do_learning(args, dataset, name):

    from .torch_utils import get_loader, get_dataset, load_model, apply_weights

    TRAIN_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'train'
    train_dataset, class_to_idx, idx_to_class = get_dataset(args, TRAIN_DATA_DIR)
    train_loader = get_loader(args, train_dataset)
    VAL_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'val'
    val_dataset, class_to_idx, idx_to_class = get_dataset(args, VAL_DATA_DIR)
    val_loader = get_loader(args, val_dataset)

    model_filename = args.data_cache / f'{name}.pth'
    json_filename = args.data_cache / model_filename.name.replace('.pth', '.json')
    lock_filename = args.data_cache / model_filename.name.replace('.pth', '.lock')


    # --- 3. Load the Pre-trained ResNet Model ---

    def touch(fname): open(fname, 'w').close()

    # %rm {lock_filename}

    df_train = None
    should_resume_training = not lock_filename.exists()

    if json_filename.exists():
        print(f"Load JSON from pre-trained resnet {json_filename}")
        df_train = pd.read_json(json_filename, orient='records')
        print(f"{model_filename}: accuracy = {df_train['acc_val'][-5:].mean():.3f}")
        should_resume_training = (df_train['epoch'].max() + 1 < args.num_epochs) and (not lock_filename.exists())

    if should_resume_training:
        touch(lock_filename) # as we do a training let's lock it
        # we need to train the model or finish a training that already started
        print(f"Training model {args.model_name}, file= {model_filename} - image_size={args.image_size}")

        model = load_model(args, model_path = model_filename if model_filename.is_file() else None)
        if args.verbose:
            num_classes = len(val_loader.dataset.classes)
            num_ftrs = model.fc.out_features
            print(f'Model has {num_ftrs} output features to final FC layer for {num_classes} classes.')

                
        start_time = time.time()
        model_retrain, df_train = train_model(args, model=model, train_loader=train_loader, val_loader=val_loader, df_train=df_train, model_filename=model_filename, json_filename=json_filename)
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    return model_filename, json_filename