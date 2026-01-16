from pathlib import Path
from .utils import get_device, set_seed
#############################################################
#############################################################
import platform
import numpy as np
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass
verbose = False

import os
USER = os.environ['USER']  # username
import platform
HOST = platform.uname()[1]

@dataclass
class Params:

    # platform-dependent variables
    if USER=='uvb28bo': # Jean Zay
        DATAROOT = Path('/lustre/fsn1/projects/rech/fsx/uvb28bo/data')
    # elif '.cluster' in HOST: # mesocentre
    #     DATAROOT = '/scratch/lperrinet/science/Deep_learning/data'
    #     num_workers = 8
        batch_size: int = 64 # Set number of images per input batch
        num_workers: int = 4
        prefetch_factor: int = 4
    elif 'm-gpu' in HOST: # MESONET
        DATAROOT = Path.home() / 'data' / 'Imagenet'
        batch_size: int = 128 # Set number of images per input batch
        num_workers: int = 4
        prefetch_factor: int = 0
        # num_workers = 16
    elif 'gaia' in HOST: # MAC STUDIO
        # batch_size = 512 # Set the batch size for training and validation
        # num_workers = 2    
        DATAROOT = Path.home() / 'data' / 'Imagenet'
        batch_size: int = 512 # Set number of images per input batch
        num_workers: int = 4
        prefetch_factor: int = 0
    else:
        DATAROOT = Path.home() / 'data' / 'Imagenet'
        batch_size: int = 32 # Set number of images per input batch
        num_workers: int = 4
        prefetch_factor: int = 0

    image_size: int = 224 # base resolution of the image (224, 224)
    grid_size: int = 224 # base resolution of the image (224, 224)
    do_mask: bool = False # Whether apply a circular mask to the image
    do_fovea: bool = False # Whether apply a log-polar transform to the image
    use_hexagonal_grid: bool = True # Whether to use hexagonal packing for the log-polar grid
    rs_min: float = -0.01 # Set minimum radius of the log-polar grid
    rs_max: float = -6.00 # Set maximum radius of the log-polar grid
    angle_start: float = -np.pi/4 # Set the intial angle for the grid
    angle_margin: float = np.pi/16 # Set a margin angle to wrap the circle
    mode: str = "bilinear"
    padding_mode: str = "zeros"
    # padding_mode: str = "border"


    # model_name: str = 'resnet50' # Name of the model to use
    model_name: str = 'convnext_base' # Name of the model to use

    # https://github.com/pytorch/vision/tree/main/references/classification#convnext
    # num_epochs: int = 1
    num_epochs: int = 200
    subset_factor: int = 1 # set for DEBUGging
    optimizer_name: str = 'adam'
    # loss_name: str = 'BCEWithLogitsLoss'
    # loss_name: str = 'NegLogitLoss'
    loss_name: str = 'CrossEntropyLoss'
    base_lr: float = 34.e-6
    final_lr: float = 1.e-6
    num_warmup_epochs: int = 20
    delta1: float = 0.&
    delta2: float = 100e-6
    weight_decay: float = 20e-6
    label_smoothing: float = 0.002 # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    do_full_training: bool = True
    do_augment: bool = True
    augment_magnitude: int = 8
    augment_proba: int = .10

    seed: int = 1998 # Set the seed for reproducibility 
    shuffle: bool = True # Whether to shuffle the data during training
    data_cache = Path('cached_data')
    figures_folder = Path('figures')
    verbose: bool = verbose

    def __post_init__(self):
        self.data_cache.mkdir(exist_ok=True)
        self.figures_folder.mkdir(exist_ok=True)
        self.device = get_device(verbose=self.verbose)
        set_seed(seed=self.seed, seed_torch=True, verbose=self.verbose)


#############################################################
#############################################################
all_model_names = ['resnet18', 'resnet50', 'resnet101'] 
all_model_names_ls = [':', '-.', '-'] 
all_model_names_color = ['blue', 'blue', 'blue']
all_cn_model_names = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']
all_cn_model_names_color = ['blue', 'blue', 'blue']
all_cn_model_names_ls = [':', '-.', '-'] 
all_datasets = ['full', 'bbox']
all_datasets_color = ['blue', 'orange']
all_datasets_ls = ['-', '-']
#############################################################
#############################################################
