from pathlib import Path
from .utils import get_device, set_seed
#############################################################
#############################################################
import platform
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass

@dataclass
class Params:
    
    DATAROOT = Path.home() / 'data'

    image_size: int = 224 # base resolution of the image (224, 224)
    do_mask: bool = True # Whether apply a circular mask to the image
    num_epochs: int = 7 # 
    n_train_stop: int = 100 # set to 1000 for DEBUGging
    # n_train_stop: int = 0 # set to zero to use all images
    seed: int = 1998 # Set the seed for reproducibility 
    batch_size: int = 64 # Set number of images per input batch
    num_workers:int = 4

    # interpolation = T.InterpolationMode.BILINEAR
    # padding_mode = "border"

    model_name: str = 'resnet50' # Name of the model to use
    do_scratch: bool = False # Whether to train from scratch (True) or use pretrained weights (False)

    batch_size = 250 # Set the batch size for training and validation
    batch_size = 64 # Set the batch size for training and validation

    num_epochs: int = 1
    lr: float = 1.e-4
    delta1: float = 0.05
    delta2: float = 0.001
    weight_decay: float = 0.0
    # label_smoothing: float = 0. # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    shuffle: bool = True # Whether to shuffle the data during training
    data_cache = Path('cached_data')
    data_cache.mkdir(exist_ok=True)
    figures_folder = Path('figures')
    figures_folder.mkdir(exist_ok=True)

    verbose: bool = True
    device = get_device(verbose=verbose)
    set_seed(seed=seed, seed_torch=True, verbose=verbose)
