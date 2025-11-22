"""Main module."""

#############################################################
#############################################################
import torch
import numpy as np
import torchvision.transforms as T

from torchvision import datasets

import torchvision.transforms as transforms

def get_idx_to_label(args):
    ##############
    LABELS_FILE = args.data_cache / 'imagenet_class_index.json' # Local cache file name

    try:
        import json # Don't forget to import json
        # Check if we already have the file
        if not LABELS_FILE.exists():
            import requests

            # --- 4. Download and Load the ImageNet Class Index (with caching) ---
            LABELS_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

            print(f"Downloading labels to {LABELS_FILE}...")
            response = requests.get(LABELS_URL)
            response.raise_for_status()
            with open(LABELS_FILE, 'w') as f:
                json.dump(response.json(), f)
        else:
            print(f"Loading labels from local cache {LABELS_FILE}...")
            
        # In both cases, load from the local file
        with open(LABELS_FILE, 'r') as f:
            class_idx = json.load(f)

        # Create a simple mapping from index to class name for easy lookup
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

        return idx2label

    except requests.exceptions.RequestException as e:
        print(f"Error downloading labels: {e}")
        exit()
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error handling local label file: {e}")
        exit()
    if args.verbose: print(f'Got a list with {len(idx2label)} labels in {LABELS_FILE} ')


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

class CleanRotations_class(object): 
    """Apply a rotation to the image.
    Clean because apply a circular mask to the image so the border does not reveal the rotation."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, images):
        temp = []
        images = images.unsqueeze(dim=0) if len(images) == 3 else images
        for image in images:
            for angle in self.angles:
                temp.append(T.functional.rotate(image, angle=angle, expand = False)) 
        return torch.stack(temp)

from .retinoto_py import get_preprocess
def get_loader(args, DATA_DIR):
    preprocess = get_preprocess(args)
    # --- 2. Create Dataset and DataLoader using ImageFolder ---
    # ImageFolder automatically infers class names from directory names
    # and maps them to integer indices.
    val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess)

    # The dataset provides a mapping from class index to class name (folder name)
    class_to_idx = val_dataset.class_to_idx
    # We often want the inverse mapping for printing results
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # The DataLoader handles batching, shuffling (for training), and loading data efficiently.
    # For evaluation, we don't need to shuffle.
    # A batch size of 1 is simplest for per-image analysis, but you can use larger batches.
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    return val_loader, class_to_idx, idx_to_class

import torchvision.models as models
def load_model(args):
    """
    Load the model from the torchvision library.
    
    """

    if args.model_name=='resnet18':
        model = models.resnet18(weights=None if args.do_scratch else models.ResNet18_Weights.DEFAULT)
    elif args.model_name=='resnet50':
        model = models.resnet50(weights=None if args.do_scratch else models.ResNet50_Weights.DEFAULT)
    elif args.model_name=='resnet101':
        model = models.resnet101(weights=None if args.do_scratch else models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f'Unknown model {args.model_name}')
    model = model.to(args.device)
    return model


def count_parameters(model):
    """Counts the total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }

def count_layers(model, layer_type=None):
    """
    Counts the layers in a PyTorch model.
    If `layer_type` is None, counts all nn.Module children.
    If `layer_type` is specified (e.g., nn.Conv2d), counts only those.
    """
    if layer_type is None:
        return sum(1 for _ in model.modules())
    else:
        return sum(1 for module in model.modules() if isinstance(module, layer_type))

#############################################################
import matplotlib.pyplot as plt
import matplotlib
import torchvision

def imgs_to_np(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std):
    images = torchvision.utils.make_grid(img_list, nrow=11)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)


def imshow(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std, 
           title:str=None, fig_height:float=7., fig=None, ax:matplotlib.axes.Axes=None, 
           fontsize=14, save:bool=False, name:str=None, dpi = 'figure', exts:list=['pdf', 'png']):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_height*len(img_list), fig_height))

    inp = imgs_to_np(img_list, im_mean=im_mean, im_std=im_std)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title != None: 
        fig.suptitle(title, fontsize=fontsize)
    fig.set_facecolor(color='white')
    #plt.tight_layout()

    if save:
        for ext in exts: fig.savefig(figures_folder / f'{name}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=0, edgecolor=None)
    else:
        return fig, ax