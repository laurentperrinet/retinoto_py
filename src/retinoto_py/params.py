from pathlib import Path
from .utils import get_device, set_seed
#############################################################
#############################################################
import platform
# https://docs.python.org/3/library/dataclasses.html?highlight=dataclass#module-dataclasses
from dataclasses import dataclass
verbose = False

@dataclass
class Params:
    
    DATAROOT = Path.home() / 'data' / 'Imagenet'

    image_size: int = 224 # base resolution of the image (224, 224)
    do_mask: bool = False # Whether apply a circular mask to the image
    do_fovea: bool = False # Whether apply a log-polar transform to the image
    rs_min: float = 0.00 # Set minimum radius of the log-polar grid
    rs_max: float = -7.50 # Set maximum radius of the log-polar grid
    padding_mode: str = "zeros"
    # padding_mode = "border"

    seed: int = 2025 # Set the seed for reproducibility 
    # batch_size: int = 64 # Set number of images per input batch
    batch_size: int  = 80 # Set the batch size for training and validation
    num_workers: int = 1
    in_memory: bool = False


    # model_name: str = 'resnet50' # Name of the model to use
    model_name: str = 'convnext_base' # Name of the model to use


    # num_epochs: int = 1
    num_epochs: int = 41
    subset_factor: int = 1 # set for DEBUGging
    lr: float = 2.e-7
    delta1: float = 0.2
    delta2: float = 0.007
    weight_decay: float = 0.003
    label_smoothing: float = 0.05 # See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

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
all_cn_model_names = ['convnext_tiny', 'convnext_base', 'convnext_large'] #'convnext_small', 
all_cn_model_names_color = ['blue', 'blue', 'blue']
all_cn_model_names_ls = [':', '-.', '-'] 
all_datasets = ['full', 'bbox']
all_datasets_color = ['blue', 'orange']
all_datasets_ls = ['-', '-']
#############################################################
#############################################################
