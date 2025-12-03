"""
Useful torch snippets to use in the main module.

"""

#############################################################
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision
import torch
from torchvision import datasets
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.nn.functional as nnf
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
# from torchvision.transforms.functional import crop
import torchvision.transforms as transforms
# from torchvision.transforms import v2 as transforms TODO use v2 !!
import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
import torch.nn as nn
#############################################################
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"iCCP: profile",
    category=UserWarning,
)


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

def get_label_to_idx(args):
    idx2label = get_idx_to_label(args)  # Get the list of labels
    label2idx = {label: idx for idx, label in enumerate(idx2label)}
    return label2idx

# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 


def torch_loader(path: str) -> torch.Tensor:
    """
    Load an image file using torchvision.io.read_image.
    The returned tensor is float32 in the range [0, 1] and has shape (C, H, W).

    Parameters
    ----------
    path : str
        Full path to the image file.

    Returns
    -------
    torch.Tensor
        Float tensor ready for torchvision transforms.
    """
    # read_image returns a UInt8 tensor (C, H, W) with values 0‑255
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Suppress warnings just for this operation
        img = read_image(path)                # still on CPU
    # Convert to float and normalise
    img = img.float() / 255.0

    # -----------------------------------------------------------------
    # 2‑channel / 1‑channel handling (most transforms expect 3 channels)
    # -----------------------------------------------------------------
    if img.shape[0] == 1:                 # grayscale → replicate to 3 channels
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:               # RGBA → drop alpha (or you could blend)
        img = img[:3, :, :]                # keep only RGB
    elif img.shape[0] not in (3,):
        raise RuntimeError(f"Unsupported number of channels ({img.shape[0]}) in {path}")

    return img

class InMemoryImageDataset(Dataset):
    """Load entire ImageFolder dataset into memory"""
    def __init__(self, root, transform=None, n_stop=0, is_valid_file=None, do_pil=True):
        # Use ImageFolder to handle directory structure and class mapping
        if do_pil:
            image_folder = datasets.ImageFolder(root=root, 
                                                is_valid_file=is_valid_file,
                                                transform=None, 
                                                )
        else:
            image_folder = datasets.ImageFolder(root=root, 
                                                loader=torch_loader,
                                                is_valid_file=is_valid_file,
                                                transform=None, 
                                                )
        
        self.class_to_idx = image_folder.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = image_folder.classes
        self.transform = transform
        
        # Load all images into memory
        # print("Loading dataset into memory...")
        self.images = []
        self.labels = []
        if n_stop==0:
            n_total = len(image_folder)
            idxs = np.arange(n_total) 
        else:
            n_total = min((n_stop, len(image_folder)))
            idxs = np.random.permutation(len(image_folder))[:n_total].astype(int)

        for idx in tqdm(idxs, desc='Putting images in memory', total=n_total, leave=False):
            self.images.append(image_folder[idx][0])
            self.labels.append(image_folder[idx][1])

        # print(f"Loaded {len(self.images)} images into memory")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label



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

class ApplyMask(object):
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

# Prefer direct module import to avoid static analysis issues in some environments
def get_grid(args, endpoint=False):
    """
    Generate a grid for the log-polar mapping
    """

    rs_ = torch.logspace(args.rs_min, args.rs_max, args.image_size, base=2) # Radial distances (log scale)
    # TODO : add a margin in angles in order to get an overrepresentation
    ts_ = torch.linspace(0, torch.pi*2, args.image_size+1)[:-1] 
    grid_xs = torch.outer(rs_, torch.cos(ts_)) # X-coordinates
    grid_ys = torch.outer(rs_, torch.sin(ts_)) # Y-coordinates	
    
    return torch.stack((grid_xs, grid_ys), 2)#.to(args.device) # (H_scaled, W_scaled, 2)

class transform_apply_grid(object): 
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    def __init__(self, logPolar_grid, padding_mode, mode):
        self.grid = logPolar_grid
        self.padding_mode = padding_mode
        self.mode = mode

    def __call__(self, images):
        result =  nnf.grid_sample(images.unsqueeze(dim=0), 
                                    self.grid.unsqueeze(dim=0), 
                                    padding_mode=self.padding_mode, align_corners=False, 
                                    mode=self.mode)
        return result.squeeze(0)

def get_preprocess(args, angle_min=None, angle_max=None, 
                   interpolation=TF.InterpolationMode.BILINEAR, mode='bilinear', do_pil=True):
    # --- 5. Define Image Pre-processing ---
    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    transform_list = []
    transform_list.append(transforms.Resize((args.image_size, args.image_size)))
    transform_list.append(transforms.CenterCrop(args.image_size))
    if do_pil: transform_list.append(transforms.ToTensor())  # Convert the image to a PyTorch Tensor
    # transform_list.append(transforms.Lambda(lambda x: TF.resize(x, args.image_size)))
    # transform_list.append(transforms.Lambda(lambda x: TF.center_crop(x, args.image_size)))
               
    # Si les deux angles ne sont pas None, on applique la rotation
    if angle_min is not None and angle_max is not None:
        transform_list.append(transforms.RandomRotation(degrees=(angle_min, angle_max), interpolation=interpolation))
    
    transform_list.append(transforms.RandomHorizontalFlip())

    if args.do_fovea: # apply log-polar mapping to the image
        grid_polar = get_grid(args)
        transform_list.append(transform_apply_grid(grid_polar, padding_mode=args.padding_mode, mode=mode))

    transform_list.append(transforms.Normalize(mean=im_mean, std=im_std))

    if args.do_mask and not(args.do_fovea):
        # Créer le masque une seule fois avec la taille de l'image
        mask = make_mask(image_size=args.image_size)
        # Ajouter notre transform personnalisée à la liste
        transform_list.append(ApplyMask(mask))

    # Créer la chaîne de prétraitement finale
    preprocess = transforms.Compose(transform_list)
    return preprocess

def get_dataset(args, DATA_DIR, angle_min=None, angle_max=None, in_memory=None, n_stop=0, do_pil=True):
    preprocess = get_preprocess(args, angle_min=angle_min, angle_max=angle_max, do_pil=do_pil)

    is_valid_file = lambda p: p.lower().endswith(('.png', '.jpg', '.jpeg'))
    # --- 2. Create Dataset and DataLoader using ImageFolder ---
    # ImageFolder automatically infers class names from directory names
    # and maps them to integer indices.
    if in_memory is None: in_memory = args.in_memory
    if in_memory:
        # Use in-memory dataset instead of ImageFolder
        dataset = InMemoryImageDataset(root=DATA_DIR, transform=preprocess, n_stop=n_stop, is_valid_file=is_valid_file, do_pil=do_pil)
    else:    
        if do_pil:
            dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess, is_valid_file=is_valid_file)
        else:
            dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess, is_valid_file=is_valid_file, loader=torch_loader)
        if n_stop>0: raise('not implemented')
    # # The dataset provides a mapping from class index to class name (folder name)
    # class_to_idx = dataset.class_to_idx
    # # We often want the inverse mapping for printing results
    # idx_to_class = {v: k for k, v in class_to_idx.items()}
    return dataset

from .utils import set_seed

import random, numpy as np, torch

def _seed_worker(seed: int):
    """
    This function will be executed **once** in each DataLoader worker
    (after the process is spawned).  It only sets the NumPy and Python
    random seeds – everything else (torch seed) is handled by the
    `generator` argument of DataLoader.
    """
    np.random.seed(seed)
    random.seed(seed)

def get_loader(args, dataset, drop_last=True, seed=None):
    # The DataLoader handles batching, shuffling (for training), and loading data efficiently.
    # For evaluation, we don't need to shuffle.
    # A batch size of 1 is simplest for per-image analysis, but you can use larger batches.
    # if seed is None: seed = args.seed
    # set_seed(seed=seed, seed_torch=True, verbose=False)
    # val_loader = DataLoader(dataset, batch_size=args.batch_size, 
    #                         shuffle=args.shuffle, drop_last=drop_last,
    #                         num_workers=args.num_workers,
    #                         pin_memory=args.in_memory,                             # unified memory – no benefit
    #                         persistent_workers=False,                     # recreate workers each epoch
    #                         # prefetch_factor=2,                            # default, keep it small                            
    #                         worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed)
    #                         )
    if seed is None:
        seed = getattr(args, "seed", 0)

    # `worker_init_fn` receives the worker id, we ignore it and just call the top‑level function.
    worker_init = lambda wid: _seed_worker(seed)   # simple closure over an int → picklable

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        # worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed),  # deterministic shuffling
        pin_memory=False,               # unified memory → no need for pinned host memory
        persistent_workers=False,       # workers are spawned each epoch (safer for transform changes)
        # prefetch_factor=2,              # a small pre‑fetch queue is enough on M‑series
    )
    return loader

import torchvision.models as models
def load_model(args, model_filename=None):
    """
    Load the model from the torchvision library.
    
    """

    if args.model_name=='resnet18':
        model = models.resnet18(weights=None if args.do_scratch else models.ResNet18_Weights.DEFAULT)
    elif args.model_name=='resnet50':
        model = models.resnet50(weights=None if args.do_scratch else models.ResNet50_Weights.DEFAULT)
    elif args.model_name=='resnet101':
        model = models.resnet101(weights=None if args.do_scratch else models.ResNet101_Weights.DEFAULT)
    elif args.model_name=='convnext_tiny':
        model = models.convnext_tiny(weights=None if args.do_scratch else models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_small':
        model = models.convnext_small(weights=None if args.do_scratch else models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_base':
        model = models.convnext_base(weights=None if args.do_scratch else models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_large':
        model = models.convnext_large(weights=None if args.do_scratch else models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'Unknown model {args.model_name}')
    model = model.to(args.device)
    
    if not model_filename is None:
        model = apply_weights(model, model_filename, args.device, verbose=args.verbose)

    return model

def apply_weights(model, model_filename, device, verbose=True):
    """
    Apply the weights to the model.
    Args:
        model: torch model, the model to apply the weights to
        model_filename: str, path to the weights file
        verbose: bool, whether to print the loading message or not
    Returns:    
        model: torch model, the model with the weights applied
        """
    if verbose: print(f'loading .... {model_filename}')
    model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)))
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

def imgs_to_np(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std):
    images = torchvision.utils.make_grid(img_list, nrow=11)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

from .utils import savefig
def imshow(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std, 
           title:str=None, fig_height:float=7., fig=None, ax:matplotlib.axes.Axes=None, 
           figures_folder:str='figures', fontsize=14, save:bool=False, name:str=None, 
           dpi = 'figure', exts:list=['pdf', 'png']):
    
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
        savefig(fig=fig, name=name, exts=exts, dpi=dpi, figures_folder=figures_folder)
    else:
        return fig, ax