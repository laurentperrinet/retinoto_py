import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path

#############################################################

def get_device(verbose):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            mps_info = f'Running on MPS device (Apple Silicon/MacOS)'
            try:
                if hasattr(torch.backends.mps, 'metal_version'):
                    mps_info += f' - metal_version = {torch.backends.mps.metal_version()}'
            except:
                pass
            mps_info += f' - macos_version = {platform.mac_ver()[0]}'
            print(mps_info)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose: print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())    
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    return device


# set seed function
def set_seed(seed=None, seed_torch:bool=True, verbose:bool=False):
  "Define a random seed or use a predefined seed for repeatability"
  if seed is None:
    seed = np.random.choice(2 ** 32)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  if verbose: print(f'Random seed {seed} has been set.')    

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f">> Scanning variables occupying GPU memory...")
    for var_name, var in globals().items():
        if torch.is_tensor(var) and var.is_cuda:
            print(f"{var_name}: {var.element_size() * var.nelement() / 1024**2:.2f} MB")

    

#############################################################
#############################################################
import platform
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

@dataclass
class Params:
    
    HOST = platform.uname()[1]
    USER = Path.home().owner() if hasattr(Path.home(), 'owner') else Path.home().name

    DATAROOT = Path.home() / 'data'

    image_size: int = 224 # base resolution of the image (224, 224)
    num_epochs: int = 5 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = 64 # Set number of images per input batch

    # interpolation = T.InterpolationMode.BILINEAR
    # padding_mode = "border"

    model_name: str = 'resnet50' # Name of the model to use
    do_scratch: bool = False # Whether to train from scratch (True) or use pretrained weights (False)

    batch_size = 50 # Set the batch size for training and validation
    batch_size = 250 # Set the batch size for training and validation
    shuffle: bool = True # Whether to shuffle the data during training
    num_workers = 2
    data_cache = Path('cached_data')
    data_cache.mkdir(exist_ok=True)
    figures_folder = Path('figures')
    figures_folder.mkdir(exist_ok=True)

    verbose: bool = True
    device = get_device(verbose=verbose)
    set_seed(seed=seed, seed_torch=True, verbose=verbose)

    if verbose: 
        print('Welcome on', platform.platform(), end='\t')
        print(f'User {USER} Working on host {HOST} with device {device}, pytorch=={torch.__version__}')


#############################################################
#############################################################

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
    
def get_loader(args, DATA_DIR):
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

def to_save(fig:matplotlib.figure.Figure, name:str, exts:list=['pdf', 'png'], folder:str='figs'):
    """Save the figure in the specified formats.
    Args:   
        fig (matplotlib.figure.Figure): Figure to save.
        name (str): Name of the file to save.
        exts (list): List of extensions to save the figure in.
        folder (str): Folder to save the figure in.
    """

fig_width = 20


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