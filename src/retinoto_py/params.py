from pathlib import Path
from .utils import get_device, set_seed
#############################################################
#############################################################
import platform
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass

@dataclass
class Params:
    
    DATAROOT = Path.home() / 'data' / 'Imagenet'

    image_size: int = 224 # base resolution of the image (224, 224)
    do_mask: bool = False # Whether apply a circular mask to the image
    do_fovea: bool = False # Whether apply a log-polar transform to the image
    rs_min: float = 0.00 # Set minimum radius of the log-polar grid
    rs_max: float = -5.00 # Set maximum radius of the log-polar grid
    padding_mode: str = "zeros"
    # padding_mode = "border"

    seed: int = 2018 # Set the seed for reproducibility 
    batch_size: int = 64 # Set number of images per input batch
    num_workers: int = 4
    in_memory: bool = True


    # model_name: str = 'resnet50' # Name of the model to use
    model_name: str = 'resnet101' # Name of the model to use
    do_scratch: bool = False # Whether to train from scratch (True) or use pretrained weights (False)

    batch_size: int  = 80 # Set the batch size for training and validation

    # num_epochs: int = 1
    num_epochs: int = 20 
    n_train_stop: int = 512*batch_size # set for DEBUGging
    n_val_stop: int = 64*batch_size # set for DEBUGging
    do_full_training: bool = True
    # n_train_stop: int = 0 # set to zero to use all images
    # n_val_stop: int = 0 # set to zero to use all images
    lr: float = 4.e-3
    delta1: float = 0.1
    delta2: float = 0.
    weight_decay: float = 0.05
    label_smoothing: float = 0.1 # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    shuffle: bool = True # Whether to shuffle the data during training
    data_cache = Path('cached_data')
    data_cache.mkdir(exist_ok=True)
    figures_folder = Path('figures')
    figures_folder.mkdir(exist_ok=True)

    verbose: bool = True
    device = get_device(verbose=verbose)
    set_seed(seed=seed, seed_torch=True, verbose=verbose)
