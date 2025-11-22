import torch
import numpy as np
import platform
from pprint import pprint
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
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass, asdict, field

@dataclass
class Params:
    


    import platform
    HOST = platform.uname()[1]
    USER = Path.home().owner() if hasattr(Path.home(), 'owner') else Path.home().name
    # DATAROOT = Path('/Volumes/SSD1TO/DeepLearningDatasets')
    # DATAROOT = Path('data')
    DATAROOT = Path.home() / 'data'
    # root: str = f'{DATAROOT}/Imagenet_{data_set_type}' # Directory containing images to perform the training
    # folders: list = field(default_factory=lambda: ['val', 'train']) # Set the training and validation folders relative to the root
    # DATAVAL = DATAROOT.join('val')
    # DATATRAIN = DATAROOT +  Path('train')
    
    image_size: int = 224 # base resolution of the image (224, 224)
    num_epochs: int = 5 # 
    n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = 64 # Set number of images per input batch

    # interpolation = T.InterpolationMode.BILINEAR
    # padding_mode = "border"

    batch_size = 50 # Set the batch size for training and validation
    batch_size = 250 # Set the batch size for training and validation
    num_workers = 2
    data_cache = Path('cached_data')
    data_cache.mkdir(exist_ok=True)

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

def get_loader(args, DATA_DIR):
    # --- 5. Define Image Pre-processing ---
    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    preprocess = transforms.Compose([
        transforms.Resize(256),                # Resize the shortest side to 256px
        transforms.CenterCrop(224),            # Crop the center 224x224 pixels
        transforms.ToTensor(),                 # Convert the image to a PyTorch Tensor
        transforms.Normalize(                  # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    return val_loader, class_to_idx, idx_to_class