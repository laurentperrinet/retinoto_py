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
